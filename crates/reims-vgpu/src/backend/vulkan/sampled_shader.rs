//! Final sampled-shader semantics shared by compute, vertex and fragment.
//! Pixel-coordinate offsets become coordinate arithmetic; composite RGB keeps
//! its native Q11 interpolation rather than floating-image filtering.

use crate::backend::vulkan::engine::{SamplerCompareFunction, SamplerResource};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refusal(&'static str);

impl Refusal {
    pub(crate) fn slug(self) -> &'static str { self.0 }
}

fn refusal(reason: &'static str) -> Refusal { Refusal(reason) }

struct Builder {
    next: u32,
    declarations: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<(u16, u32, Vec<u32>)>,
    constants: HashMap<(u32, Vec<u32>), u32>,
    ext: u32,
    instructions: Vec<u32>,
}

impl Builder {
    fn id(&mut self) -> u32 { let id = self.next; self.next += 1; id }

    fn instruction(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
        out.push((((operands.len() + 1) as u32) << 16) | u32::from(opcode));
        out.extend_from_slice(operands);
    }

    fn ty(&mut self, opcode: u16, operands: &[u32]) -> u32 {
        if let Some((_, id, _)) = self.types.iter().find(|(op, _, args)|
            *op == opcode && args == operands)
        { return *id; }
        let id = self.id();
        let mut args = vec![id];
        args.extend_from_slice(operands);
        Self::instruction(&mut self.declarations, opcode, &args);
        self.types.push((opcode, id, operands.to_vec()));
        id
    }

    fn constant(&mut self, ty: u32, words: &[u32], composite: bool) -> u32 {
        if let Some(id) = self.constants.get(&(ty, words.to_vec())) { return *id; }
        let id = self.id();
        let mut args = vec![ty, id];
        args.extend_from_slice(words);
        Self::instruction(&mut self.declarations, if composite { 44 } else { 43 }, &args);
        self.constants.insert((ty, words.to_vec()), id);
        id
    }

    fn op(&mut self, opcode: u16, ty: u32, operands: &[u32]) -> u32 {
        let id = self.id();
        self.result(opcode, ty, id, operands);
        id
    }

    fn result(&mut self, opcode: u16, ty: u32, id: u32, operands: &[u32]) {
        let mut args = vec![ty, id];
        args.extend_from_slice(operands);
        Self::instruction(&mut self.instructions, opcode, &args);
        if matches!(opcode, 129 | 131 | 133 | 142) {
            Self::instruction(&mut self.annotations, 71, &[id, 42]);
        }
    }

    fn ext(&mut self, ty: u32, operation: u32, operands: &[u32]) -> u32 {
        let mut args = vec![self.ext, operation];
        args.extend_from_slice(operands);
        self.op(12, ty, &args)
    }

    fn pixel_sample(
        &mut self, instruction: &[u32], dimension: u32,
        value_types: &HashMap<u32, u32>, zeros: &HashSet<u32>,
    ) -> Result<Vec<u32>, Refusal> {
        let float = self.ty(22, &[32]);
        let coordinate_type = if dimension == 0 { float } else { self.ty(23, &[float, 2]) };
        let mut coordinate = instruction[4];
        if value_types.get(&coordinate) != Some(&coordinate_type) {
            return Err(refusal("vulkan_sampler_coordinate_type"));
        }
        let opcode = instruction[0] as u16;
        let mask = instruction.get(5).copied().unwrap_or(0);
        if mask & !31 != 0 || mask & 24 == 24 {
            return Err(refusal("vulkan_sampler_image_operands"));
        }
        let mut position = 6;
        let mut kept = Vec::new();
        let mut kept_mask = 0;
        for (bit, count) in [(1, 1), (2, 1), (4, 2), (8, 1), (16, 1)] {
            if mask & bit == 0 { continue; }
            let args = instruction.get(position..position + count)
                .ok_or_else(|| refusal("vulkan_sampler_image_operands"))?;
            position += count;
            if bit == 1 {
                if !zeros.contains(&args[0]) {
                    return Err(refusal("vulkan_sampler_lod_bias"));
                }
            } else if bit == 8 || bit == 16 {
                if zeros.contains(&args[0]) { continue; }
                let offset_type = value_types.get(&args[0])
                    .ok_or_else(|| refusal("vulkan_sampler_offset_type"))?;
                let component = if dimension == 0 { *offset_type } else {
                    self.types.iter().find_map(|(op, id, args)|
                        (*op == 23 && id == offset_type && args.len() == 2 && args[1] == 2)
                            .then(|| args[0])
                    ).ok_or_else(|| refusal("vulkan_sampler_offset_type"))?
                };
                let signed = self.types.iter().find_map(|(op, id, args)|
                    (*op == 21 && *id == component && args.len() == 2 && args[0] == 32)
                        .then(|| args[1] != 0)
                ).ok_or_else(|| refusal("vulkan_sampler_offset_type"))?;
                // An offset is in texels. For pixel coordinates its exact
                // spelling is a coordinate add, not a forbidden image operand.
                let delta = self.op(if signed { 111 } else { 112 }, coordinate_type, args);
                coordinate = self.op(129, coordinate_type, &[coordinate, delta]);
            } else {
                kept_mask |= bit;
                kept.extend_from_slice(args);
            }
        }
        if position.min(instruction.len()) != instruction.len() {
            return Err(refusal("vulkan_sampler_image_operands"));
        }
        if opcode == 87 {
            if kept_mask != 0 { return Err(refusal("vulkan_sampler_image_operands")); }
            kept_mask = 2;
            kept.push(self.constant(float, &[0], false));
        } else if kept_mask != 2 && kept_mask != 4 {
            return Err(refusal("vulkan_sampler_lod_operands"));
        }
        let mut args = vec![instruction[1], instruction[2], instruction[3], coordinate, kept_mask];
        args.extend(kept);
        let mut sample = Vec::new();
        Self::instruction(&mut sample, 88, &args);
        Ok(sample)
    }

    fn linear(
        &mut self, result_type: u32, result: u32, image: u32, coordinate: u32,
        normalized: bool, offset: Option<u32>, value_types: &HashMap<u32, u32>,
    ) -> Result<(), Refusal> {
        let float = self.ty(22, &[32]);
        let f2 = self.ty(23, &[float, 2]);
        let f4 = self.ty(23, &[float, 4]);
        if result_type != f4 { return Err(refusal("vulkan_planar_shader_sample_type")); }
        let int = self.ty(21, &[32, 1]);
        let i2 = self.ty(23, &[int, 2]);
        let zero = self.constant(int, &[0], false);
        let one = self.constant(int, &[1], false);
        let izero = self.constant(i2, &[zero, zero], true);
        let ione = self.constant(i2, &[one, one], true);
        let fx1 = self.constant(float, &[1.0f32.to_bits()], false);
        let half = self.constant(float, &[0.5f32.to_bits()], false);
        let half2 = self.constant(f2, &[half, half], true);
        let half4 = self.constant(f4, &[half; 4], true);
        let scale = self.constant(float, &[2048.0f32.to_bits()], false);
        let scale4 = self.constant(f4, &[scale; 4], true);
        let reciprocal = self.constant(float, &[(1.0f32 / 2048.0).to_bits()], false);
        let reciprocal4 = self.constant(f4, &[reciprocal; 4], true);
        let size = self.op(103, i2, &[image, zero]);
        let position = if normalized {
            let extent = self.op(111, f2, &[size]);
            self.op(133, f2, &[coordinate, extent])
        } else { coordinate };
        let position = self.op(131, f2, &[position, half2]);
        let floor = self.ext(f2, 8, &[position]);
        let fraction = self.op(131, f2, &[position, floor]);
        let x = self.op(81, float, &[fraction, 0]);
        let y = self.op(81, float, &[fraction, 1]);
        let ix = self.op(131, float, &[fx1, x]);
        let iy = self.op(131, float, &[fx1, y]);
        let mut origin = self.op(110, i2, &[floor]);
        if let Some(offset) = offset {
            let offset_type = value_types.get(&offset)
                .ok_or_else(|| refusal("vulkan_planar_shader_offset_type"))?;
            let component = self.types.iter().find_map(|(op, id, args)|
                (*op == 23 && id == offset_type && args.len() == 2 && args[1] == 2)
                    .then(|| args[0])
            ).ok_or_else(|| refusal("vulkan_planar_shader_offset_type"))?;
            if !self.types.iter().any(|(op, id, args)|
                *op == 21 && *id == component && args.len() == 2 && args[0] == 32)
            { return Err(refusal("vulkan_planar_shader_offset_type")); }
            let offset = if *offset_type == i2 { offset } else { self.op(124, i2, &[offset]) };
            // Image offsets shift integer taps, not normalized coordinates:
            // adding offset/size to UV would perturb the fractional weights.
            origin = self.op(128, i2, &[origin, offset]);
        }
        let maximum = self.op(130, i2, &[size, ione]);
        let mut total = None;
        for (dx, dy, wx, wy) in [(zero, zero, ix, iy), (one, zero, x, iy),
            (zero, one, ix, y), (one, one, x, y)]
        {
            let offset = self.constant(i2, &[dx, dy], true);
            let point = self.op(128, i2, &[origin, offset]);
            let point = self.ext(i2, 45, &[point, izero, maximum]);
            let texel = self.op(95, f4, &[image, point, 2, zero]);
            let codes = self.op(133, f4, &[texel, scale4]);
            let weight = self.op(133, float, &[wx, wy]);
            let weighted = self.op(142, f4, &[codes, weight]);
            total = Some(match total {
                Some(prior) => self.op(129, f4, &[prior, weighted]),
                None => weighted,
            });
        }
        let rounded = self.op(129, f4, &[total.expect("four texels"), half4]);
        let rounded = self.ext(f4, 8, &[rounded]);
        self.result(133, f4, result, &[rounded, reciprocal4]);
        Ok(())
    }
}

pub(crate) fn specialize(
    words: &[u32], q11_bindings: &[u32], samplers: &[SamplerResource],
) -> Result<Vec<u32>, Refusal> {
    let pixel_samplers: HashSet<u32> = samplers.iter()
        .filter(|sampler| sampler.unnormalized_coordinates).map(|sampler| sampler.binding).collect();
    if q11_bindings.is_empty() && pixel_samplers.is_empty() { return Ok(words.to_vec()); }
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(refusal("vulkan_planar_shader_module"));
    }
    let mut instructions = Vec::new();
    let mut position = 5;
    while position < words.len() {
        let length = (words[position] >> 16) as usize;
        if length == 0 || length > words.len() - position {
            return Err(refusal("vulkan_planar_shader_module"));
        }
        instructions.push(&words[position..position + length]);
        position += length;
    }
    let mut builder = Builder {
        next: words[3], declarations: Vec::new(), annotations: Vec::new(), types: Vec::new(),
        constants: HashMap::new(), ext: 0, instructions: Vec::new(),
    };
    let mut bindings = HashMap::new();
    let mut origins = HashMap::new();
    let mut value_types = HashMap::new();
    let mut zeros = HashSet::new();
    let mut combined = HashMap::new();
    let mut has_query_capability = false;
    for instruction in &instructions {
        let opcode = instruction[0] as u16;
        match opcode {
            11 if instruction.len() >= 3 => {
                let name: Vec<u8> = instruction[2..].iter().flat_map(|v| v.to_le_bytes()).collect();
                if name.starts_with(b"GLSL.std.450\0") { builder.ext = instruction[1]; }
            }
            17 if instruction.get(1) == Some(&50) => has_query_capability = true,
            19..=39 if instruction.len() >= 2 => {
                builder.types.push((opcode, instruction[1], instruction[2..].to_vec()));
            }
            43 | 44 if instruction.len() >= 4 => {
                builder.constants.insert((instruction[1], instruction[3..].to_vec()), instruction[2]);
                if (opcode == 43 && instruction[3..].iter().all(|word| *word == 0))
                    || (opcode == 43 && instruction.len() == 4 && instruction[3] == 0x8000_0000
                        && builder.types.iter().any(|(op, id, args)|
                            *op == 22 && *id == instruction[1] && args == &[32]))
                    || (opcode == 44 && instruction[3..].iter().all(|id| zeros.contains(id)))
                { zeros.insert(instruction[2]); }
            }
            46 if instruction.len() == 3 => { zeros.insert(instruction[2]); }
            71 if instruction.len() == 4 && instruction[2] == 33 => {
                bindings.insert(instruction[1], instruction[3]);
            }
            _ => {}
        }
        if instruction.len() >= 3 && !matches!(opcode, 19..=39 | 71 | 72)
            && builder.types.iter().any(|(_, id, _)| *id == instruction[1])
        { value_types.insert(instruction[2], instruction[1]); }
    }
    let new_import = builder.ext == 0;
    if new_import { builder.ext = builder.id(); }
    let mut replacements = HashMap::new();
    let mut q11_linear_rewritten = false;
    let tracked = |binding: &u32| q11_bindings.contains(binding) || pixel_samplers.contains(binding);
    for (index, instruction) in instructions.iter().enumerate() {
        let opcode = instruction[0] as u16;
        match opcode {
            61 | 83 if instruction.len() >= 4 => {
                let source = instruction[3];
                if let Some(binding) = bindings.get(&source).or_else(|| origins.get(&source)) {
                    origins.insert(instruction[2], *binding);
                }
                if let Some(sampled) = combined.get(&source).copied() {
                    combined.insert(instruction[2], sampled);
                }
                if opcode == 83 && zeros.contains(&source) { zeros.insert(instruction[2]); }
                value_types.insert(instruction[2], instruction[1]);
            }
            100 if instruction.len() == 4 => {
                if let Some((image, _)) = combined.get(&instruction[3]) {
                    if let Some(binding) = origins.get(image).copied() {
                        origins.insert(instruction[2], binding);
                    }
                }
                value_types.insert(instruction[2], instruction[1]);
            }
            65..=67 if instruction.len() >= 4 => {
                if bindings.get(&instruction[3]).is_some_and(tracked) {
                    return Err(refusal("vulkan_sampled_shader_descriptor_array"));
                }
            }
            86 if instruction.len() == 5 => {
                if let Some(sampler) = origins.get(&instruction[4]) {
                    combined.insert(instruction[2], (instruction[3], *sampler));
                } else if origins.get(&instruction[3]).is_some_and(|b| q11_bindings.contains(b)) {
                    return Err(refusal("vulkan_planar_shader_sampler_origin"));
                }
            }
            87 | 88 if instruction.len() >= 5 => {
                let Some(&(image, sampler_binding)) = combined.get(&instruction[3]) else { continue; };
                let q11 = origins.get(&image).is_some_and(|b| q11_bindings.contains(b));
                let pixel = pixel_samplers.contains(&sampler_binding);
                if !q11 && !pixel { continue; }
                let sampler = samplers.iter().find(|s| s.binding == sampler_binding)
                    .ok_or_else(|| refusal("vulkan_planar_shader_sampler_missing"))?;
                if q11 && (sampler.min_filter != sampler.mag_filter || sampler.min_filter > 1
                    || sampler.max_anisotropy > 1 || sampler.compare_function != SamplerCompareFunction::Never
                ) {
                    return Err(refusal("vulkan_planar_sampler_filter"));
                }
                let image_type = value_types.get(&image)
                    .and_then(|id| builder.types.iter().find(|(op, tid, _)|
                        *op == 25 && tid == id)).map(|(_, _, args)| args);
                if !image_type.is_some_and(|args| args.len() >= 7
                    && (args[1] == 1 || (!q11 && args[1] == 0))
                    && args[3] == 0 && args[4] == 0 && args[5] == 1)
                {
                    return Err(refusal("vulkan_sampled_shader_image_shape"));
                }
                let sample = if pixel {
                    let dimension = image_type.expect("checked image shape")[1];
                    builder.pixel_sample(instruction, dimension, &value_types, &zeros)?
                } else { instruction.to_vec() };
                if !q11 || sampler.min_filter == 0 {
                    if sample != *instruction || !builder.instructions.is_empty() {
                        builder.instructions.extend(sample);
                        replacements.insert(index, std::mem::take(&mut builder.instructions));
                    }
                    continue;
                }
                if sampler.address_mode_u != 0 || sampler.address_mode_v != 0 {
                    return Err(refusal("vulkan_planar_sampler_address"));
                }
                let mask = sample.get(5).copied().unwrap_or(0);
                if mask & !(1 | 2 | 4 | 8 | 16) != 0 || mask & 24 == 24 {
                    return Err(refusal("vulkan_planar_shader_image_operands"));
                }
                let offset = if mask & 24 != 0 {
                    let index = 6 + usize::from(mask & 1 != 0) + usize::from(mask & 2 != 0)
                        + 2 * usize::from(mask & 4 != 0);
                    Some(*sample.get(index)
                        .ok_or_else(|| refusal("vulkan_planar_shader_image_operands"))?)
                } else { None };
                builder.linear(sample[1], sample[2], image, sample[4],
                    !sampler.unnormalized_coordinates, offset, &value_types)?;
                q11_linear_rewritten = true;
                replacements.insert(index, std::mem::take(&mut builder.instructions));
            }
            89..=94 | 96 | 97 if instruction.len() >= 4 => {
                if combined.get(&instruction[3]).is_some_and(|(image, sampler)|
                    origins.get(image).is_some_and(|b| q11_bindings.contains(b))
                        || pixel_samplers.contains(sampler))
                {
                    return Err(refusal("vulkan_sampled_shader_image_operation"));
                }
            }
            57 | 62 | 80 | 82 | 169 | 245 | 254 | 400 => {
                let start = if matches!(opcode, 62 | 254) { 1 } else { 3 };
                if instruction.iter().skip(start).any(|id|
                    bindings.get(id).or_else(|| origins.get(id)).is_some_and(tracked)
                    || combined.get(id).is_some_and(|(image, sampler)|
                        origins.get(image).is_some_and(|b| q11_bindings.contains(b))
                            || pixel_samplers.contains(sampler)))
                {
                    return Err(refusal("vulkan_sampled_shader_alias"));
                }
            }
            _ => {}
        }
    }
    if replacements.is_empty() { return Ok(words.to_vec()); }
    let mut output = words[..5].to_vec();
    output[3] = builder.next;
    if q11_linear_rewritten && !has_query_capability { Builder::instruction(&mut output, 17, &[50]); }
    let mut inserted_import = !new_import || !q11_linear_rewritten;
    let mut inserted_declarations = false;
    let mut inserted_annotations = false;
    for (index, instruction) in instructions.into_iter().enumerate() {
        let opcode = instruction[0] as u16;
        if !inserted_annotations && (19..=39).contains(&opcode) {
            output.extend_from_slice(&builder.annotations);
            inserted_annotations = true;
        }
        if !inserted_import && opcode == 14 {
            let mut import = vec![builder.ext];
            import.extend(b"GLSL.std.450\0\0\0\0".chunks_exact(4).map(|c|
                u32::from_le_bytes(c.try_into().expect("four bytes"))));
            Builder::instruction(&mut output, 11, &import);
            inserted_import = true;
        }
        if !inserted_declarations && opcode == 54 {
            output.extend_from_slice(&builder.declarations);
            inserted_declarations = true;
        }
        if let Some(replacement) = replacements.get(&index) {
            output.extend_from_slice(replacement);
        } else {
            output.extend_from_slice(instruction);
        }
    }
    if !inserted_import || !inserted_declarations {
        return Err(refusal("vulkan_planar_shader_module"));
    }
    Ok(output)
}

#[cfg(test)]
#[path = "sampled_shader/graphics_tests.rs"]
mod graphics_tests;

#[cfg(test)]
#[path = "sampled_shader/probe_tests.rs"]
mod probe_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn module(model: u32) -> Vec<u32> {
        let mut out = vec![0x0723_0203, 0x0001_0000, 0, 32, 0];
        let mut op = |code, args: &[u32]| Builder::instruction(&mut out, code, args);
        op(17, &[1]);
        op(14, &[0, 1]);
        op(15, &[model, 20, 0x6e69_616d, 0]);
        if model == 5 { op(16, &[20, 17, 1, 1, 1]); }
        if model == 4 { op(16, &[20, 7]); }
        for (variable, binding) in [(13, 32), (14, 160)] {
            op(71, &[variable, 34, 0]);
            op(71, &[variable, 33, binding]);
        }
        op(71, &[26, 3]);
        op(72, &[26, 0, 35, 0]);
        op(71, &[29, 34, 0]);
        op(71, &[29, 33, 0]);
        op(19, &[1]);
        op(22, &[2, 32]);
        op(23, &[3, 2, 2]);
        op(23, &[4, 2, 4]);
        op(21, &[5, 32, 1]);
        op(25, &[6, 2, 1, 0, 0, 0, 1, 0]);
        op(26, &[7]);
        op(27, &[8, 6]);
        op(32, &[9, 0, 6]);
        op(32, &[10, 0, 7]);
        op(33, &[12, 1]);
        op(30, &[26, 4]);
        op(32, &[27, 2, 26]);
        op(32, &[28, 2, 4]);
        op(59, &[27, 29, 2]);
        op(43, &[5, 30, 0]);
        op(59, &[9, 13, 0]);
        op(59, &[10, 14, 0]);
        op(43, &[2, 15, 0.5f32.to_bits()]);
        op(44, &[3, 16, 15, 15]);
        op(43, &[2, 17, 0]);
        op(54, &[1, 20, 0, 12]);
        op(248, &[18]);
        op(61, &[6, 21, 13]);
        op(61, &[7, 22, 14]);
        op(86, &[8, 23, 21, 22]);
        op(88, &[4, 24, 23, 16, 2, 17]);
        op(65, &[28, 31, 29, 30]);
        op(62, &[31, 24]);
        op(253, &[]);
        op(56, &[]);
        out
    }

    fn offset_module(model: u32, offset: [i32; 2], bias: Option<f32>) -> Vec<u32> {
        let input = module(model);
        let mut output = input[..5].to_vec();
        output[3] = 37;
        let mut position = 5;
        while position < input.len() {
            let length = (input[position] >> 16) as usize;
            let instruction = &input[position..position + length];
            if instruction[0] as u16 == 54 {
                Builder::instruction(&mut output, 23, &[32, 5, 2]);
                Builder::instruction(&mut output, 43, &[5, 33, offset[0] as u32]);
                Builder::instruction(&mut output, 43, &[5, 34, offset[1] as u32]);
                Builder::instruction(&mut output, 44, &[32, 35, 33, 34]);
                Builder::instruction(&mut output, 43, &[2, 36, bias.unwrap_or(0.0).to_bits()]);
            }
            if instruction[0] as u16 == 88 {
                if bias.is_some() {
                    Builder::instruction(&mut output, 87, &[4, 24, 23, 16, 1 | 8, 36, 35]);
                } else {
                    Builder::instruction(&mut output, 88, &[4, 24, 23, 16, 2 | 8, 17, 35]);
                }
            } else { output.extend_from_slice(instruction); }
            position += length;
        }
        output
    }

    fn instructions(words: &[u32]) -> Vec<&[u32]> {
        let mut out = Vec::new();
        let mut position = 5;
        while position < words.len() {
            let length = (words[position] >> 16) as usize;
            out.push(&words[position..position + length]);
            position += length;
        }
        out
    }

    #[test]
    fn pixel_offsets_are_coordinate_adds_not_forbidden_sampler_operands_in_every_stage() {
        let normalized = SamplerResource::normalized_default(160);
        let mut pixel = normalized.clone();
        pixel.unnormalized_coordinates = true;
        for model in [0, 4, 5] {
            for offset in [[0, 0], [1, -2]] {
                let input = offset_module(model, offset, None);
                assert_eq!(specialize(&input, &[], &[normalized.clone()]).unwrap(), input);
                let output = specialize(&input, &[], &[pixel.clone()]).unwrap();
                let instructions = instructions(&output);
                let sample = instructions.iter().find(|i| i[0] as u16 == 88).unwrap();
                assert_eq!(sample[5], 2, "only explicit LOD remains");
                assert_eq!(sample.len(), 7);
                assert_eq!(instructions.iter().any(|i| i[0] as u16 == 111), offset != [0, 0]);
                assert_eq!(crate::runtime::spirv_bind::validate(&output),
                    crate::runtime::spirv_bind::SpirvValidation::Accepted);
            }
        }
    }

    #[test]
    fn pixel_zero_bias_becomes_explicit_level_zero_and_nonzero_bias_refuses() {
        let mut sampler = SamplerResource::normalized_default(160);
        sampler.unnormalized_coordinates = true;
        for bias in [0.0, -0.0] {
            let output = specialize(&offset_module(4, [0, 0], Some(bias)), &[], &[sampler.clone()]).unwrap();
            let instructions = instructions(&output);
            assert!(instructions.iter().all(|i| i[0] as u16 != 87));
            let sample = instructions.iter().find(|i| i[0] as u16 == 88).unwrap();
            assert_eq!(sample[5], 2);
            assert_eq!(crate::runtime::spirv_bind::validate(&output),
                crate::runtime::spirv_bind::SpirvValidation::Accepted);
        }
        assert_eq!(specialize(&offset_module(4, [0, 0], Some(1.0)), &[], &[sampler]),
            Err(refusal("vulkan_sampler_lod_bias")));
    }

    #[test]
    fn dynamic_pixel_offsets_use_their_value_type_without_needing_a_constant() {
        let mut input = offset_module(5, [1, -2], None);
        input[3] = 38;
        let mut output = input[..5].to_vec();
        for instruction in instructions(&input) {
            if instruction[0] as u16 == 88 {
                Builder::instruction(&mut output, 83, &[32, 37, 35]);
                Builder::instruction(&mut output, 88, &[4, 24, 23, 16, 2 | 16, 17, 37]);
            } else { output.extend_from_slice(instruction); }
        }
        let mut sampler = SamplerResource::normalized_default(160);
        sampler.unnormalized_coordinates = true;
        let lowered = specialize(&output, &[], &[sampler]).unwrap();
        let instructions = instructions(&lowered);
        assert!(instructions.iter().any(|i| i[0] as u16 == 111 && i[3] == 37));
        assert!(instructions.iter().filter(|i| i[0] as u16 == 88).all(|i| i[5] == 2));
        assert_eq!(crate::runtime::spirv_bind::validate(&lowered),
            crate::runtime::spirv_bind::SpirvValidation::Accepted);
    }

    #[test]
    fn pixel_offsets_and_planar_q11_interpolation_are_one_specialization() {
        let mut sampler = SamplerResource::normalized_default(160);
        sampler.unnormalized_coordinates = true;
        let output = specialize(&offset_module(5, [1, -2], None), &[32], &[sampler]).unwrap();
        let instructions = instructions(&output);
        assert_eq!(instructions.iter().filter(|i| i[0] as u16 == 95).count(), 4);
        assert!(instructions.iter().all(|i| !matches!(i[0] as u16, 87 | 88)));
        assert_eq!(crate::runtime::spirv_bind::validate(&output),
            crate::runtime::spirv_bind::SpirvValidation::Accepted);
    }

    #[test]
    fn normalized_planar_offsets_shift_integer_taps_without_changing_filter_weights() {
        let sampler = SamplerResource::normalized_default(160);
        for model in [0, 4, 5] {
            let output = specialize(&offset_module(model, [1, -2], None),
                &[32], &[sampler.clone()]).unwrap();
            let instructions = instructions(&output);
            assert_eq!(instructions.iter().filter(|i| i[0] as u16 == 95).count(), 4);
            assert!(instructions.iter().any(|i| i[0] as u16 == 128 && i[4] == 35));
            assert!(instructions.iter().all(|i| i[0] as u16 != 136));
            assert_eq!(crate::runtime::spirv_bind::validate(&output),
                crate::runtime::spirv_bind::SpirvValidation::Accepted);
        }
    }

    #[test]
    fn planar_linear_specialization_validates_for_compute_vertex_and_fragment() {
        let sampler = SamplerResource::normalized_default(160);
        for model in [0, 4, 5] {
            let input = module(model);
            let output = specialize(&input, &[32], &[sampler.clone()]).unwrap();
            assert!(output.len() > input.len());
            let mut index = 5;
            let mut fetches = 0;
            while index < output.len() {
                fetches += usize::from(output[index] as u16 == 95);
                index += (output[index] >> 16) as usize;
            }
            assert_eq!(fetches, 4);
            assert_eq!(crate::runtime::spirv_bind::validate(&output),
                crate::runtime::spirv_bind::SpirvValidation::Accepted);
        }
    }

    #[test]
    fn nonplanar_and_nearest_modules_are_unchanged_and_unproven_filters_refuse() {
        let input = module(5);
        assert_eq!(specialize(&input, &[], &[]).unwrap(), input);
        let mut sampler = SamplerResource::normalized_default(160);
        sampler.min_filter = 0; sampler.mag_filter = 0;
        assert_eq!(specialize(&input, &[32], &[sampler.clone()]).unwrap(), input);
        sampler.mag_filter = 1;
        assert_eq!(specialize(&input, &[32], &[sampler.clone()]),
            Err(refusal("vulkan_planar_sampler_filter")));
        sampler.min_filter = 1; sampler.address_mode_u = 2;
        assert_eq!(specialize(&input, &[32], &[sampler]),
            Err(refusal("vulkan_planar_sampler_address")));
    }

    #[test]
    fn copied_sampled_images_keep_specialization_and_memory_forwarding_refuses() {
        let input = module(5);
        let mut copied = input[..5].to_vec();
        copied[3] = 33;
        let mut position = 5;
        while position < input.len() {
            let length = (input[position] >> 16) as usize;
            let instruction = &input[position..position + length];
            if instruction[0] as u16 == 88 {
                Builder::instruction(&mut copied, 83, &[8, 32, 23]);
                let mut sample = instruction.to_vec();
                sample[3] = 32;
                copied.extend(sample);
            } else { copied.extend_from_slice(instruction); }
            position += length;
        }
        let sampler = SamplerResource::normalized_default(160);
        let output = specialize(&copied, &[32], &[sampler.clone()]).unwrap();
        assert!(output.len() > copied.len());
        assert_eq!(crate::runtime::spirv_bind::validate(&output),
            crate::runtime::spirv_bind::SpirvValidation::Accepted);
        // An opaque image escaping through memory needs a pointer-use proof;
        // never silently let that alias bypass Q11 sample specialization.
        Builder::instruction(&mut copied, 62, &[99, 21]);
        assert_eq!(specialize(&copied, &[32], &[sampler]),
            Err(refusal("vulkan_sampled_shader_alias")));
    }

    #[test]
    fn translated_static_sampler_keeps_planar_specialization_after_fragment_relocation() {
        use crate::runtime::spirv_bind;
        let air = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/air/render_frag_static_sampler.air");
        let scratch = std::path::PathBuf::from(format!(
            "target/planar-static-sampler-{}", std::process::id(),
        ));
        std::fs::create_dir_all(&scratch).unwrap();
        let translated = metal2vulkan::translate_reflected(
            air.to_str().unwrap(), metal2vulkan::passes::Stage::Fragment, &scratch,
        );
        std::fs::remove_dir_all(&scratch).unwrap();
        let (bytes, reflection) = translated.expect("translate authored static-sampler fixture");
        let mut words: Vec<u32> = bytes.chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        spirv_bind::widen_sampled_bands(&mut words);
        assert_eq!(spirv_bind::offset_fragment_sampled_resource_bindings(&mut words), 2);
        let texture = spirv_bind::reflected_texture_descriptor(&reflection, 0).unwrap();
        let reflected = spirv_bind::reflected_sampler_descriptors(&reflection, true);
        assert_eq!(reflected.len(), 1);
        assert!(!reflected[0].guest_supplied(), "constexpr state is not a guest table slot");
        let sampler = crate::runtime::draw::vulkan::reflected_static_sampler_resource(
            "fragment", reflected[0].binding, reflected[0].static_state().unwrap(),
        ).unwrap();
        let output = specialize(&words,
            &[texture.binding + spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET],
            &[sampler]).unwrap();
        assert!(output.len() > words.len());
        assert_eq!(spirv_bind::validate(&output), spirv_bind::SpirvValidation::Accepted);
    }

    #[test]
    #[ignore = "requires an exclusive Vulkan GPU slot"]
    fn vulkan_gpu_composite_native_oracles_and_q11_filter_ties() {
        use crate::backend::vulkan::engine::*;
        use crate::protocol::planar::{BackingFormat, SampleFormat};
        use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        for format in [SampleFormat::Ycbcr10_420TwoPlane, SampleFormat::Rgb10_420TwoPlane] {
            for backing in [BackingFormat::VideoRange, BackingFormat::FullRange] {
                for filter in [0, 1] {
                    let image = crate::backend::vulkan::planar::tests::image(format, backing);
                    let expected = if format == SampleFormat::Rgb10_420TwoPlane {
                        [768.0 / 1023.0, 512.0 / 1023.0, 640.0 / 1023.0, 1.0]
                    } else if backing == BackingFormat::VideoRange {
                        [1867.0 / 2048.0, 529.0 / 2048.0, 1566.0 / 2048.0, 1.0]
                    } else { [1742.0 / 2048.0, 571.0 / 2048.0, 1479.0 / 2048.0, 1.0] };
                    let mut sampler = SamplerResource::normalized_default(160);
                    sampler.min_filter = filter; sampler.mag_filter = filter;
                    let request = ComputeRequest {
                        spirv: specialize(&module(5),
                            if format == SampleFormat::Ycbcr10_420TwoPlane { &[32] } else { &[] },
                            &[sampler.clone()]).unwrap(),
                        entry: "main".into(), dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
                        storage_buffers: vec![ComputeBufferResource {
                            binding: 0, bytes: vec![0xff; 16], writable: true,
                        }],
                        sampled_images: vec![ComputeSampledImageResource {
                            binding: 32, array_element: 0, descriptor_count: 1,
                            format: image.engine_format(), width: image.width, height: image.height,
                            mip_levels: 1, source: ComputeSampledSource::Bytes(image.bytes),
                        }],
                        samplers: vec![sampler], storage_images: vec![],
                    };
                    let output = execute_compute_request(&state, &request).unwrap();
                    let values: Vec<f32> = output.buffers[0].bytes.chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                    assert_eq!(values.len(), 4);
                    for (actual, expected) in values.iter().zip(expected) {
                        assert!((actual - expected).abs() < 1e-6,
                            "{format:?} {backing:?} filter={filter}: {values:?}");
                        if format == SampleFormat::Ycbcr10_420TwoPlane { assert_eq!(*actual, expected); }
                    }
                    eprintln!("planar compute PASS {format:?} {backing:?} filter={filter}: {values:?}");
                }
            }
        }
        let sampler = SamplerResource::normalized_default(160);
        let pixels: Vec<u8> = [0u16, 1, 2, 3].into_iter().flat_map(|q|
            [q, q, q, 2048].into_iter().flat_map(|q|
                crate::protocol::planar::sampling::q11_half(q).to_le_bytes())
        ).collect();
        let request = ComputeRequest {
            spirv: specialize(&module(5), &[32], &[sampler.clone()]).unwrap(),
            entry: "main".into(), dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
            storage_buffers: vec![ComputeBufferResource { binding: 0, bytes: vec![0xff; 16], writable: true }],
            sampled_images: vec![ComputeSampledImageResource {
                binding: 32, array_element: 0, descriptor_count: 1,
                format: StorageImageFormat::Rgba16Float, width: 2, height: 2, mip_levels: 1,
                source: ComputeSampledSource::Bytes(pixels.clone()),
            }],
            samplers: vec![sampler], storage_images: vec![],
        };
        let output = execute_compute_request(&state, &request).unwrap();
        let got = f32::from_le_bytes(output.buffers[0].bytes[..4].try_into().unwrap());
        assert_eq!(got, 2.0 / 2048.0, "Q11 interpolation rounds half up");
        for q11 in [false, true] {
            let mut sampler = SamplerResource::normalized_default(160);
            sampler.unnormalized_coordinates = true;
            let request = ComputeRequest {
                spirv: specialize(&offset_module(5, [1, 0], None),
                    if q11 { &[32] } else { &[] }, &[sampler.clone()]).unwrap(),
                entry: "main".into(), dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
                storage_buffers: vec![ComputeBufferResource { binding: 0, bytes: vec![0xff; 16], writable: true }],
                sampled_images: vec![ComputeSampledImageResource {
                    binding: 32, array_element: 0, descriptor_count: 1,
                    format: StorageImageFormat::Rgba16Float, width: 2, height: 2, mip_levels: 1,
                    source: ComputeSampledSource::Bytes(pixels.clone()),
                }],
                samplers: vec![sampler], storage_images: vec![],
            };
            let output = execute_compute_request(&state, &request).unwrap();
            let got = f32::from_le_bytes(output.buffers[0].bytes[..4].try_into().unwrap());
            assert_eq!(got, 1.0 / 2048.0, "pixel offset advances one texel, q11={q11}");
        }
        let sampler = SamplerResource::normalized_default(160);
        let pixels = [0u16, 0, 0, 4, 0, 0, 0, 4].into_iter().flat_map(|q|
            [q, q, q, 2048].into_iter().flat_map(|q|
                crate::protocol::planar::sampling::q11_half(q).to_le_bytes())
        ).collect();
        let request = ComputeRequest {
            spirv: specialize(&offset_module(5, [1, 0], None), &[32], &[sampler.clone()]).unwrap(),
            entry: "main".into(), dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
            storage_buffers: vec![ComputeBufferResource { binding: 0, bytes: vec![0xff; 16], writable: true }],
            sampled_images: vec![ComputeSampledImageResource {
                binding: 32, array_element: 0, descriptor_count: 1,
                format: StorageImageFormat::Rgba16Float, width: 4, height: 2, mip_levels: 1,
                source: ComputeSampledSource::Bytes(pixels),
            }],
            samplers: vec![sampler], storage_images: vec![],
        };
        let output = execute_compute_request(&state, &request).unwrap();
        let got = f32::from_le_bytes(output.buffers[0].bytes[..4].try_into().unwrap());
        assert_eq!(got, 2.0 / 2048.0, "normalized offset shifts taps, not fractional weights");
    }
}
