from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse


from api.services.detectors.video_service import detect_video

router = APIRouter()


@router.post("/video")
async def detect_video_api(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.05, le=1.0),
):

    tmp_path = f"runs/tmp_{file.filename}"

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    output_path = detect_video(tmp_path, conf)

    output_path = str(output_path)

    def iterfile():
        with open(output_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": "attachment; filename=detected.mp4"
        }
    )
