pub(super) fn byte_spans(pages: &[u64], page_bytes: u64) -> Option<Vec<(u64, u64)>> {
    if pages.is_empty() || !page_bytes.is_power_of_two() {
        return None;
    }
    let mut spans = pages
        .iter()
        .map(|&page| {
            if !page.is_multiple_of(page_bytes) {
                return None;
            }
            Some((page, page.checked_add(page_bytes)?))
        })
        .collect::<Option<Vec<_>>>()?;
    spans.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        if let Some(previous) = merged.last_mut().filter(|previous| start <= previous.1) {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    Some(merged)
}

pub(super) fn overlap_bytes(a: &[(u64, u64)], b: &[(u64, u64)]) -> Option<u64> {
    let (mut i, mut j, mut bytes) = (0, 0, 0u64);
    while i < a.len() && j < b.len() {
        let start = a[i].0.max(b[j].0);
        let end = a[i].1.min(b[j].1);
        bytes = bytes.checked_add(end.saturating_sub(start))?;
        if a[i].1 <= b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(bytes)
}
