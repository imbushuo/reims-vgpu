use std::collections::BTreeMap;

use super::BindingSig;
use ash::vk;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Unproven {
    Malformed,
    Version,
    Addressing,
    IndirectDecoration,
    Extension,
    UnknownInstruction,
    DescriptorSet,
}

impl crate::observe::Decline for Unproven {
    fn slug(&self) -> &'static str {
        match self {
            Self::Malformed => "storage_descriptor_proof_malformed",
            Self::Version => "storage_descriptor_proof_version",
            Self::Addressing => "storage_descriptor_proof_addressing",
            Self::IndirectDecoration => "storage_descriptor_proof_indirect_decoration",
            Self::Extension => "storage_descriptor_proof_extension",
            Self::UnknownInstruction => "storage_descriptor_proof_unknown_instruction",
            Self::DescriptorSet => "storage_descriptor_proof_descriptor_set",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Declaration coverage, not liveness: even an unused declared resource is kept.
/// An incomplete answer can never authorize dropping a descriptor.
pub(crate) struct DeclarationProof(Result<Box<[u32]>, Unproven>);

impl DeclarationProof {
    pub(in crate::backend::vulkan::engine) fn of_final_module(words: &[u32]) -> Self {
        Self(direct_declarations(words))
    }

    pub(in crate::backend::vulkan::engine) fn complete_bindings(&self) -> Option<&[u32]> {
        self.0.as_deref().ok()
    }

    #[cfg(test)]
    pub(crate) fn for_test(bindings: &[u32]) -> Self {
        Self::of_final_module(&tests::module(bindings))
    }

    #[cfg(test)]
    pub(crate) fn unproven_for_test() -> Self {
        Self(Err(Unproven::Malformed))
    }

    pub(super) fn report_unproven(&self, digest: super::Digest128) {
        if let Err(reason) = &self.0 {
            crate::observe::Emit::decline("vk_storage_descriptor_admission", reason)
                .field("shader", format!("{:016x}{:016x}", digest.a, digest.b))
                .field("bytes", digest.len)
                .field("retained", "legacy_descriptors")
                .fail();
        }
    }
}

/// Borrowed from the two retained native shader records; no per-draw module walk.
pub(crate) struct StorageDescriptorAdmission<'a> {
    vertex: &'a DeclarationProof,
    fragment: &'a DeclarationProof,
}

impl<'a> StorageDescriptorAdmission<'a> {
    pub(crate) fn new(vertex: &'a DeclarationProof, fragment: &'a DeclarationProof) -> Self {
        Self { vertex, fragment }
    }

    pub(crate) fn retains(&self, binding: u32) -> bool {
        match (&self.vertex.0, &self.fragment.0) {
            (Ok(vertex), Ok(fragment)) => {
                vertex.binary_search(&binding).is_ok() || fragment.binary_search(&binding).is_ok()
            }
            _ => true,
        }
    }

    pub(crate) fn filter_layout(&self, bindings: &mut Vec<BindingSig>) {
        bindings.retain(|binding| {
            self.retains_descriptor(
                binding.binding,
                vk::DescriptorType::from_raw(binding.ty as i32),
            )
        });
    }

    pub(crate) fn retains_descriptor(&self, binding: u32, ty: vk::DescriptorType) -> bool {
        !matches!(
            ty,
            vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::SAMPLED_IMAGE
        ) || self.retains(binding)
    }

    pub(crate) fn filter_writes(
        &self,
        writes: &mut Vec<super::super::pools::PushDescriptorBinding>,
    ) {
        use super::super::pools::PushDescriptorBinding;
        writes.retain(|write| match write {
            PushDescriptorBinding::Buffer { binding, ty, .. }
            | PushDescriptorBinding::Image { binding, ty, .. } => {
                self.retains_descriptor(*binding, *ty)
            }
        });
    }
}

fn direct_declarations(words: &[u32]) -> Result<Box<[u32]>, Unproven> {
    if words.len() < 5 || words[0] != 0x0723_0203 || words[3] == 0 || words[4] != 0 {
        return Err(Unproven::Malformed);
    }
    if !(0x0001_0000..=0x0001_0600).contains(&words[1]) || words[1] & 0xff != 0 {
        return Err(Unproven::Version);
    }
    let bound = words[3];
    let valid_id = |id: u32| id != 0 && id < bound;
    let mut decorations: BTreeMap<u32, (Option<u32>, Option<u32>)> = BTreeMap::new();
    let mut pointers = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut memory_model = false;
    let mut shader = false;
    let mut entry = false;
    let mut in_function = false;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        let end = at.checked_add(count).ok_or(Unproven::Malformed)?;
        if count == 0 || end > words.len() {
            return Err(Unproven::Malformed);
        }
        let args = &words[at + 1..end];
        match opcode {
            10 => return Err(Unproven::Extension), // OpExtension
            11 => {
                if in_function || args.len() < 2 || !valid_id(args[0]) {
                    return Err(Unproven::Malformed);
                }
                let name: Vec<_> = args[1..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                let end = name
                    .iter()
                    .position(|byte| *byte == 0)
                    .ok_or(Unproven::Malformed)?;
                if &name[..end] != b"GLSL.std.450" {
                    return Err(Unproven::Extension);
                }
            }
            73..=75 | 332 | 5632 | 5633 => return Err(Unproven::IndirectDecoration),
            14 => {
                if in_function || memory_model || args.len() != 2 {
                    return Err(Unproven::Malformed);
                }
                if args != [0, 1] {
                    return Err(Unproven::Addressing);
                }
                memory_model = true;
            }
            15 => {
                if in_function || args.len() < 3 || !valid_id(args[1]) {
                    return Err(Unproven::Malformed);
                }
                entry = true;
            }
            17 => {
                if in_function || args.len() != 1 {
                    return Err(Unproven::Malformed);
                }
                shader |= args[0] == 1;
            }
            32 => {
                // OpTypePointer
                if in_function || args.len() != 3 || !valid_id(args[0]) || !valid_id(args[2]) {
                    return Err(Unproven::Malformed);
                }
                if pointers.insert(args[0], args[1]).is_some() {
                    return Err(Unproven::Malformed);
                }
            }
            54 => {
                if in_function || args.len() != 4 {
                    return Err(Unproven::Malformed);
                }
                in_function = true;
            }
            56 => {
                if !in_function || !args.is_empty() {
                    return Err(Unproven::Malformed);
                }
                in_function = false;
            }
            59 => {
                // OpVariable
                if !(3..=4).contains(&args.len()) || !valid_id(args[0]) || !valid_id(args[1]) {
                    return Err(Unproven::Malformed);
                }
                if in_function {
                    if args[2] != 7 {
                        return Err(Unproven::Malformed);
                    }
                } else if variables.insert(args[1], (args[0], args[2])).is_some() {
                    return Err(Unproven::Malformed);
                }
            }
            71 => {
                // OpDecorate: Binding and DescriptorSet are literal, direct decorations.
                if in_function || args.len() < 2 || !valid_id(args[0]) {
                    return Err(Unproven::Malformed);
                }
                if matches!(args[1], 33 | 34) {
                    if args.len() != 3 {
                        return Err(Unproven::Malformed);
                    }
                    let pair = decorations.entry(args[0]).or_default();
                    let slot = if args[1] == 33 {
                        &mut pair.0
                    } else {
                        &mut pair.1
                    };
                    if slot
                        .replace(args[2])
                        .is_some_and(|previous| previous != args[2])
                    {
                        return Err(Unproven::Malformed);
                    }
                }
            }
            72 => {
                if in_function || args.len() < 3 || matches!(args[2], 33 | 34) {
                    return Err(Unproven::Malformed);
                }
            }
            // Unknown global/declaration instructions cannot establish complete coverage.
            _ if !in_function
                && !matches!(opcode, 0..=8 | 11 | 16 | 19..=31 | 33..=52 | 317 | 330 | 331) =>
            {
                return Err(Unproven::UnknownInstruction);
            }
            _ if opcode > 366 => return Err(Unproven::UnknownInstruction),
            _ => {}
        }
        at = end;
    }
    if in_function || !memory_model || !shader || !entry {
        return Err(Unproven::Malformed);
    }
    let mut bindings = Vec::new();
    for (&id, &(binding, set)) in &decorations {
        let (Some(binding), Some(set)) = (binding, set) else {
            return Err(Unproven::Malformed);
        };
        if set != 0 {
            return Err(Unproven::DescriptorSet);
        }
        let &(pointer, storage) = variables.get(&id).ok_or(Unproven::Malformed)?;
        if !matches!(storage, 0 | 2 | 12) || pointers.get(&pointer) != Some(&storage) {
            return Err(Unproven::Malformed);
        }
        bindings.push(binding);
    }
    for (&id, &(_, storage)) in &variables {
        if matches!(storage, 0 | 2 | 12) && !decorations.contains_key(&id) {
            return Err(Unproven::Malformed);
        }
    }
    bindings.sort_unstable();
    bindings.dedup();
    Ok(bindings.into_boxed_slice())
}

#[cfg(test)]
mod tests;
