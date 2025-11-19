"""
Author: Nathaniel Brewer

DTO for the meta data coming from(and going to) our mongo db
"""
from pydantic import BaseModel, Field # type: ignore

class AudioVersionDTO(BaseModel):
    id: str
    size: str
    sample_rate: int | None = None
    bit_rate: str | None = None
    bit_depth: str | None = None
    file_path: str

class TrackMetaDTO(BaseModel):
    id: str = Field(alias="_id") # Allows mongo _id to be inserted here 
    name: str
    length: str

    lossless: AudioVersionDTO
    lossy: AudioVersionDTO

    class Config:
        populate_by_name = True