import json

from .track_meta_dto import TrackMetaDTO
from .track_manager import TrackManager

class TrackMetaManager:
    # No-args 
    def __init__(self, data):
        self.meta_dto = self.json_to_meta_dto(data)
        self.track_manager = TrackManager(self.meta_dto.lossy.file_path, self.meta_dto.lossless.file_path)

    """
        All Json methods
    """

    def _serialize(self, to_json):
        return json.dumps(to_json, indent=4) 
    
    def _deserialize(self, from_json):
        return json.loads(from_json)

    def json_to_meta_dto(self, data):
        try:
            return TrackMetaDTO(**data)
        except Exception as e:
            print(f"Error creating DTO: {e}")
            raise
    
    def meta_dto_to_json(self, dto):
        return self._serialize(dto)
    
    """
        File Management
    """

    def get_stream(self, quality):
        if quality == "lossy":
            return self.track_manager.get_lossy_stream_url()
        else:
            return self.track_manager.get_lossless_stream_url()

    def generate_headers(self, quality):
        version = getattr(self.meta_dto, quality, None)
        if version is None:
            raise ValueError(f"Unknown quality: {quality}")

        return {
            "X-Track-Name": self.meta_dto.name,
            "X-Track-Length": self.meta_dto.length,
            "X-Sample-Rate": str(version.sample_rate),
            "X-Bit-Depth": str(version.bit_depth),
            "X-Bit-Rate": str(version.bit_rate),
            "X-Track-ID": str(version.id)
        }

    def __str__(self):
        return (
            f"TrackMetaManager("
            f"name='{self.meta_dto.name}', "
            f"id='{self.dto.id}', "
            f"length='{self.meta_dto.length}', "
            f"sample_rate={self.meta_dto.sample_rate}, "
            f"bit_depth='{self.meta_dto.bit_depth}'"
            f")"
        )