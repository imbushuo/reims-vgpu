use super::*;

pub(crate) fn module(bindings: &[u32]) -> Vec<u32> {
    let mut words = vec![
        0x0723_0203,
        0x0001_0300,
        0,
        1024,
        0,
        (2 << 16) | 17,
        1,
        (3 << 16) | 14,
        0,
        1,
        (5 << 16) | 15,
        4,
        1,
        0x6e69_616d,
        0,
        (4 << 16) | 32,
        3,
        12,
        2,
    ];
    for (index, &binding) in bindings.iter().enumerate() {
        let id = 10 + index as u32;
        words.extend([
            (4 << 16) | 71,
            id,
            33,
            binding,
            (4 << 16) | 71,
            id,
            34,
            0,
            (4 << 16) | 59,
            3,
            id,
            12,
        ]);
    }
    words
}

#[test]
fn storage_descriptor_direct_declarations_keep_even_unreferenced_buffers() {
    let proof = DeclarationProof::of_final_module(&module(&[0, 680, 672, 0]));
    assert_eq!(&*proof.0.unwrap(), &[0, 672, 680]);
}

#[test]
fn storage_descriptor_admission_unions_stages_without_dropping_real_bindings() {
    let vertex = DeclarationProof::of_final_module(&module(&[0, 1, 2, 3, 4]));
    let fragment =
        DeclarationProof::of_final_module(&module(&[672, 673, 674, 675, 676, 677, 679, 680]));
    let plan = StorageDescriptorAdmission::new(&vertex, &fragment);
    for binding in [0, 1, 2, 3, 4, 672, 673, 674, 675, 676, 677, 679, 680] {
        assert!(plan.retains(binding));
    }
    for binding in [5, 678, 681] {
        assert!(!plan.retains(binding));
    }
    let other_stage = DeclarationProof::of_final_module(&module(&[5, 681]));
    let plan = StorageDescriptorAdmission::new(&vertex, &other_stage);
    assert!(plan.retains(5) && plan.retains(681));
}

#[test]
fn storage_descriptor_grouped_declarations_retain_the_legacy_layout() {
    for instruction in [
        vec![(2 << 16) | 73, 100],
        vec![(3 << 16) | 74, 100, 10],
        vec![(4 << 16) | 75, 100, 10, 0],
        vec![(4 << 16) | 332, 10, 33, 100],
    ] {
        let mut words = module(&[0]);
        words.extend(instruction);
        let grouped = DeclarationProof::of_final_module(&words);
        assert_eq!(grouped.0, Err(Unproven::IndirectDecoration));
        let known = DeclarationProof::of_final_module(&module(&[]));
        assert!(StorageDescriptorAdmission::new(&known, &grouped).retains(681));
        assert!(StorageDescriptorAdmission::new(&grouped, &known).retains(5));
        assert_unproven_keeps_sampled_layout_and_writes(&grouped);
    }
}

#[test]
fn storage_descriptor_unknown_function_or_extended_instruction_set_preserves_bindings() {
    let mut unknown_body = module(&[]);
    unknown_body.extend([(5 << 16) | 54, 2, 1, 0, 4, (1 << 16) | 999, (1 << 16) | 56]);
    assert_eq!(
        DeclarationProof::of_final_module(&unknown_body).0,
        Err(Unproven::UnknownInstruction)
    );
    let mut unknown_set = module(&[]);
    unknown_set.extend([(3 << 16) | 11, 20, 0]);
    assert_eq!(
        DeclarationProof::of_final_module(&unknown_set).0,
        Err(Unproven::Extension)
    );
    let mut glsl = module(&[]);
    glsl.extend([(6 << 16) | 11, 20, 0x4c53_4c47, 0x6474_732e, 0x3035_342e, 0]);
    assert!(DeclarationProof::of_final_module(&glsl).0.is_ok());
}

#[test]
fn storage_descriptor_malformed_unknown_and_unsupported_proofs_never_authorize_omission() {
    let original = module(&[0]);
    let mut fixtures = vec![vec![], vec![0; 5], original[..original.len() - 1].to_vec()];
    for tail in [
        vec![0],
        vec![(6 << 16) | 71, 10],
        vec![(1 << 16) | 65535],
        vec![(2 << 16) | 10, 0],
        vec![(4 << 16) | 71, 10, 34, 1],
        vec![(4 << 16) | 71, 10, 33, 5],
    ] {
        let mut words = original.clone();
        words.extend(tail);
        fixtures.push(words);
    }
    let mut physical = original.clone();
    physical[8] = 5348;
    fixtures.push(physical);
    let mut future = original.clone();
    future[1] = 0x0001_0700;
    fixtures.push(future);
    let known = DeclarationProof::of_final_module(&module(&[]));
    for words in fixtures {
        let unknown = DeclarationProof::of_final_module(&words);
        assert!(unknown.0.is_err(), "{words:?}");
        assert!(StorageDescriptorAdmission::new(&known, &unknown).retains(678));
        assert_unproven_keeps_sampled_layout_and_writes(&unknown);
    }
}

#[test]
fn storage_descriptor_proof_requires_direct_typed_variables_and_set_zero() {
    let original = module(&[0]);
    for (at, value) in [(17, 2), (26, 1), (30, 7)] {
        let mut words = original.clone();
        words[at] = value;
        assert!(DeclarationProof::of_final_module(&words).0.is_err(), "{at}");
    }
}

#[test]
fn storage_descriptor_filter_preserves_all_other_descriptor_classes() {
    let empty = DeclarationProof::of_final_module(&module(&[]));
    let plan = StorageDescriptorAdmission::new(&empty, &empty);
    let mut layout: Vec<_> = [
        vk::DescriptorType::STORAGE_BUFFER,
        vk::DescriptorType::SAMPLED_IMAGE,
        vk::DescriptorType::SAMPLER,
        vk::DescriptorType::INPUT_ATTACHMENT,
        vk::DescriptorType::STORAGE_IMAGE,
    ]
    .into_iter()
    .map(|ty| BindingSig {
        binding: 5,
        ty: ty.as_raw() as u32,
        stages: 17,
        count: 1,
    })
    .collect();
    let expected = layout[2..].to_vec();
    plan.filter_layout(&mut layout);
    assert_eq!(layout, expected);
}

fn assert_unproven_keeps_sampled_layout_and_writes(unknown: &DeclarationProof) {
    use super::super::super::pools::PushDescriptorBinding;
    let known = DeclarationProof::of_final_module(&module(&[]));
    for admission in [
        StorageDescriptorAdmission::new(unknown, &known),
        StorageDescriptorAdmission::new(&known, unknown),
    ] {
        let mut layout = vec![BindingSig {
            binding: 704,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: 17,
            count: 2,
        }];
        let mut writes: Vec<_> = (0..2)
            .map(|array_element| PushDescriptorBinding::Image {
                binding: 704,
                array_element,
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                sampler: vk::Sampler::null(),
                view: vk::ImageView::null(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .collect();
        admission.filter_layout(&mut layout);
        admission.filter_writes(&mut writes);
        assert_eq!(layout.len(), 1);
        assert_eq!(writes.len(), 2);
    }
}
