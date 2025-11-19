import json
from bson import ObjectId

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
            # Create a copy to avoid modifying the original data
            processed_data = data.copy()
            
            # Convert ObjectId to string for _id field
            if '_id' in processed_data and isinstance(processed_data['_id'], ObjectId):
                processed_data['_id'] = str(processed_data['_id'])
            
            # Add missing id fields to lossless and lossy if they don't exist
            if 'lossless' in processed_data and 'id' not in processed_data['lossless']:
                # Generate an ID based on the _id and type
                base_id = processed_data.get('_id', 'unknown')
                processed_data['lossless']['id'] = f"{base_id}_lossless"
            
            if 'lossy' in processed_data and 'id' not in processed_data['lossy']:
                # Generate an ID based on the _id and type
                base_id = processed_data.get('_id', 'unknown')
                processed_data['lossy']['id'] = f"{base_id}_lossy"
            
            return TrackMetaDTO(**processed_data)
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
        elif quality == "lossless":
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
        # Return compact one-line JSON representation by default
        try:
            return self.toString()
        except Exception:
            return f"TrackMetaManager(name='{getattr(self.meta_dto, 'name', None)}', id='{getattr(self.meta_dto, 'id', None)}')"

    def toString(self) -> str:
        """Return the `meta_dto` as a compact, single-line JSON string.

        Uses the Pydantic model's `.json()` to preserve aliases (e.g. `_id`) and
        emits a minimal representation with no extra whitespace so it's one line.
        """
        # Prefer Pydantic's json() when available
        try:
            return self.meta_dto.json(by_alias=True, exclude_none=True, separators=(",", ":"))
        except Exception:
            # Fallback to stdlib json with dict conversion
            try:
                return json.dumps(self.meta_dto.dict(by_alias=True, exclude_none=True), separators=(",", ":"))
            except Exception as e:
                # As a last resort, return a readable str
                return str(self.meta_dto)
    
    