from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Dict, List
import logging

from .hybrid_extractor import HybridExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RequestText(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description='Текст запроса')

class ExtractionResponse(BaseModel):
    response: Dict[str, List[str]]

class HealthResponse(BaseModel):
    status: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting up')
    if not hasattr(app.state, 'extractor'):
        app.state.extractor = HybridExtractor()
        logger.info('Extractor intialized')
    yield
    logger.info('Shutting down...')

app = FastAPI(
    title='Hybrid NER Extractor',
    description='Extract entities using spaCy and fine-tuned LLM',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_headers=['*'],
    allow_methods=['*']
)

@app.get('/health', response_model=HealthResponse, summary='Проверка здоровья')
async def health_check():
    return HealthResponse(status='ok')

@app.post('/extract', response_model=ExtractionResponse, summary='Извлечь сущности')
async def extract_entities(request: RequestText):
    """
    Принимает текст, возвращает извлечённые сущности в виде словаря.
    """
    extractor = app.state.extractor
    if extractor is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        entities = extractor.extract(request.text)
        logger.info(f'Extracted entities from text: {request.text[:50]}')
        return ExtractionResponse(response=entities)
    except Exception as e:
        logger.error(f'Extraction error: {str(e)}')
        raise HTTPException(status_code=500, detail='Internal server error')