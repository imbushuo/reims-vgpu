use super::*;
use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
use crate::runtime::host::{FakeHost, MemError};

#[test]
fn cpu_snapshot_audit_reuses_density_fold_and_disagreement_contract() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.gather_witness.audit = AuditDensity::EveryBind;
    let mut host = FakeHost::new();
    let key = GatherKey::TaskBuffer { task_id: 1, gva: 4 };
    let pages = [0x4000];
    let mut reads = 0;
    for _ in 0..4 {
        note_cpu_read(&mut state, &mut host, key, &pages, 17, 0x4000, |_| {
            reads += 1;
            Ok(vec![0x22; 17])
        })
        .unwrap();
    }
    assert_eq!(
        reads, 2,
        "rearm and stride precede the existing seed/compare"
    );
    let seen = note_cpu_read(&mut state, &mut host, key, &pages, 17, 0x4000, |_| {
        Ok(vec![0x33; 17])
    })
    .unwrap();
    assert!(
        !seen.cpu_read_vouched(),
        "audit disagreement spends the generation"
    );
    assert_eq!(seen.audit_bytes, 17);
    let next = note_cpu_read(&mut state, &mut host, key, &pages, 17, 0x4000, |_| {
        Ok(vec![0x33; 17])
    })
    .unwrap();
    assert!(next.cpu_read_vouched());
    assert_eq!(seen.identity, next.identity);
}

#[test]
fn cpu_snapshot_failed_or_short_audits_fail_closed_and_can_recover() {
    for short in [false, true] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.gather_witness.audit = AuditDensity::EveryBind;
        let mut host = FakeHost::new();
        let key = GatherKey::TaskBuffer { task_id: 1, gva: 4 };
        let pages = [0x4000];
        let mut previous = None;
        for _ in 0..3 {
            previous = Some(
                note_cpu_read(&mut state, &mut host, key, &pages, 17, 0x4000, |_| {
                    Ok(vec![0x22; 17])
                })
                .unwrap(),
            );
        }
        let error = note_cpu_read(&mut state, &mut host, key, &pages, 17, 0x4000, |_| {
            if short {
                Ok(vec![0; 16])
            } else {
                Err(MemError::Unmapped)
            }
        });
        assert!(error.is_err());
        let next = note_cpu_read(&mut state, &mut host, key, &pages, 17, 0x4000, |_| {
            Ok(vec![0x22; 17])
        })
        .unwrap();
        assert!(!next.cpu_read_vouched());
        assert_ne!(next.identity, previous.unwrap().identity);
    }
}
