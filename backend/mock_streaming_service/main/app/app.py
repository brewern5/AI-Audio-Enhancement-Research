from typing import Union

from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

from fastapi import FastAPI, APIRouter, HTTPException, Query, Body  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
import uvicorn  # type: ignore

from ..exceptions.exceptions import QueryException

from ..db.mongo.db import Db
from ..db.track.dto.track_meta_manager import TrackMetaManager

app = FastAPI()
router = APIRouter()

try:
    db = Db()
except Exception as e:
    print("Db not created!")
    print(e)


"""
    Get methods
"""


@app.get("/tracks")
async def get_track(name: Union[str, None] = Query(default=None, description="Name of the track")):

    if name is None:
        raise HTTPException(
            status_code=400, detail="Query parameter 'name' is required"
        )
    try:
        result = db.get_from_collection("Metadata", name)

        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Query: '{name}' was not found.")

@app.get("/tracks/stream/lossy")
async def stream_lossy_track(name: Union[str, None] = Query(default=None, description="Name of the track")):
    if name is None:
        raise HTTPException(
            status_code=400, detail="Query parameter 'name' is required"
        )

    try:
        # Get from db
        result = db.get_from_collection("Metadata", name)

        # store as DTO
        dto_manager = TrackMetaManager(result)

        headers = dto_manager.generate_headers('lossy')

        # using a generator since we want a constant flow of chunks to the client
        def generate():

            # Chunk out audio
            for chunk in dto_manager.get_stream("lossy"):
                yield chunk
        
        # Begin Stream
        return StreamingResponse(generate(), media_type="audio/mpeg", headers=headers)
    except QueryException as qe:
        raise HTTPException(status_code=404, detail=f"Query: '{name}' was not found.")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error!")
        

@app.get("/tracks/stream/lossless")
async def stream_lossless_track(name: Union[str, None] = Query(default=None, description="Name of the track")):
    if name is None:
        raise HTTPException(
            status_code=400, detail="Query parameter 'name' is required"
        )

    try:
        
        """Process the audio"""

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Query: '{name}' was not found.")



"""
    Post Methods
"""


@app.post("/tracks/new")
async def post_new(item: dict = Body(...)):
    
    try:
        # Create DTO
        dto_item = TrackMetaManager(item)

        # Pass to Db to add item
        posted = db.post_to_collection("Metadata", dto_item)

        return {"status": "success", "message": "Track added successfully"}
        
    except Exception as e:
        print(f"Error in post_new: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Error processing request: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("main.app.app:app", host="0.0.0.0", port=8000, reload=True)
