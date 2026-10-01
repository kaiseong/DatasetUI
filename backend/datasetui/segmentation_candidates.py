"""Validate persisted candidate identities against immutable sample jobs."""
import hashlib
from datasetui.segmentation_frames import read_snapshot_frame, _safe_regular_path
from datasetui.segmentation_selection import SegmentationGuidanceError


def validate_candidate_references(database, settings, spec, profile_id):
    """Candidates bind to their own detection frame, not to the frame being viewed."""
    for prompt in spec.prompts:
        for ref in prompt.selected_candidates:
            job = database.get_job(str(ref.sample_id))
            source = job['payload'].get('spec', {})
            if (job['kind'] != 'segmentation.sample' or job['status'] != 'succeeded'
                or job['profile_id'] != profile_id
                or any(str(source.get(k)) != str(getattr(spec, k)) for k in ('dataset_id', 'episode_index', 'video_key'))):
                raise SegmentationGuidanceError('다른 원본·프로필의 탐지 후보는 사용할 수 없습니다. 후보를 다시 찾으세요.')
            if (job['result'] or {}).get('tracked_clip'):
                raise SegmentationGuidanceError('추적 샘플의 후보는 사용할 수 없습니다. 후보 찾기로 다시 탐지하세요.')
            if prompt.frame_index != source['frame_index']:
                raise SegmentationGuidanceError(
                    f"객체 {prompt.object_id or ''}의 탐지 후보 프레임이 Instruction 프레임과 다릅니다. 후보를 다시 찾으세요.")
            original = next((p for p in source.get('prompts', [])
                             if p.get('object_id') == prompt.object_id and p.get('text') == prompt.text), None)
            candidate = next((c for c in job['result'].get('candidates', []) if c['candidate_id'] == ref.candidate_id), None)
            if not original or not candidate or candidate.get('object_id') != prompt.object_id or candidate.get('detection_score') is None:
                raise SegmentationGuidanceError(
                    f"객체 {prompt.object_id or ''}: Instruction 문구가 바뀌었거나 후보가 없습니다. 후보를 다시 찾아 선택하세요.")
            if candidate['detection_score'] < (prompt.confidence_threshold or 0):
                raise SegmentationGuidanceError('선택한 후보가 최소 탐지 점수보다 낮습니다. 기준을 낮추거나 후보를 다시 고르세요.')
            provenance = job['result'].get('model_provenance', {})
            from datasetui.sam3_engine import MIXED_HINT_POLICY
            if provenance.get('checkpoint_sha256') != settings.sam3_checkpoint_sha256 or provenance.get('hint_policy') not in {MIXED_HINT_POLICY, 'instance-box-points-detection-score-v2', 'instance-box-points-grounding-identity-v3'}:
                raise SegmentationGuidanceError('모델이 변경되었습니다. 후보를 다시 탐지하세요.')
            read_snapshot_frame(database, settings, str(spec.dataset_id), source['frame_token'],
                                spec.episode_index, spec.video_key, source['frame_index'], verify_only=True)
            root = settings.jobs_root / 'segmentation-samples' / str(ref.sample_id)
            artifact = candidate['artifact_name']
            path = _safe_regular_path(root, root / artifact)
            if hashlib.sha256(path.read_bytes()).hexdigest() != job['result']['artifacts'].get(artifact):
                raise SegmentationGuidanceError('탐지 후보 파일이 변경되었습니다. 후보를 다시 찾으세요.')

            grounding = root / f"grounding-{ref.candidate_id}.png"
            if grounding.exists() or grounding.name in job['result']['artifacts']:
                path = _safe_regular_path(root, grounding)
                if hashlib.sha256(path.read_bytes()).hexdigest() != job['result']['artifacts'].get(grounding.name):
                    raise SegmentationGuidanceError('탐지 후보 원본 파일이 변경되었습니다. 후보를 다시 찾으세요.')
