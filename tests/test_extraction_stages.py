from tools.summarize_extraction_stages import stages, summarize


def test_upstream_failure_does_not_invent_output_detection_failure():
    assert stages(False, 'source_or_reference_embedding_failed') == (None, None, False)
    assert stages(True, 'source_or_reference_embedding_failed') == (True, True, True)
    counts = summarize([True, False, None])
    assert counts['success_rate'] is None
    assert counts['success_rate_lower_bound'] == 1/3
    assert counts['success_rate_upper_bound'] == 2/3


def test_failure_stage_preserves_prior_success_and_empty_denominator():
    assert stages(False, 'alignment_failed') == (True, False, False)
    assert stages(False, 'invalid_embedding') == (True, True, False)
    assert stages(False, 'detection_error:ValueError') == (False, False, False)
    assert summarize([])['success_rate'] is None
