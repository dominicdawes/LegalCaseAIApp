"""
LightRAG Integration for Existing RAG Pipeline
Enhances naive vector RAG with knowledge graph capabilities using DGraph
"""

import os
import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

# LightRAG and graph libraries
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.embedding import openai_embedding

# DGraph integration
import pydgraph
from pydgraph import DgraphClient, DgraphClientStub

# Your existing imports
from utils.llm_clients.llm_factory import LLMFactory
from utils.supabase_utils import supabase_client
from tasks.database import get_db_connection, get_redis_connection

logger = logging.getLogger(__name__)

# ——— Configuration ———————————————————————————————————————————————

DGRAPH_HOST = os.getenv("DGRAPH_HOST", "localhost:9080")
DGRAPH_ALPHA = os.getenv("DGRAPH_ALPHA", "localhost:9080")

# LightRAG settings
LIGHTRAG_WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ——— Data Structures ——————————————————————————————————————————————

@dataclass
class EntityExtraction:
    name: str
    entity_type: str
    description: str
    confidence_score: float
    source_chunks: List[str]
    name_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None

@dataclass
class RelationshipExtraction:
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    weight: float
    confidence_score: float
    source_evidence: List[str]
    description_embedding: Optional[List[float]] = None

# ——— DGraph Client Manager ———————————————————————————————————————————

class DGraphManager:
    """
    DGraph Cloud client for knowledge graph operations
    Handles entity/relationship CRUD and graph queries
    """
    
    def __init__(self):
        self.client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize DGraph client connection"""
        if self._initialized:
            return
            
        try:
            # Create DGraph client
            client_stub = DgraphClientStub(DGRAPH_ALPHA)
            self.client = DgraphClient(client_stub)
            
            # Set up schema
            await self._setup_schema()
            
            self._initialized = True
            logger.info("✅ DGraph client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ DGraph initialization failed: {e}")
            raise
    
    async def _setup_schema(self):
        """Set up DGraph schema for knowledge graph"""
        # This would apply the schema from the previous artifact
        # In practice, you'd load this from a file or define it here
        schema = """
        # The schema content from the previous artifact would go here
        # This is typically done once during setup
        """
        
        try:
            operation = pydgraph.Operation(schema=schema)
            await self.client.alter(operation)
            logger.info("✅ DGraph schema applied successfully")
        except Exception as e:
            logger.error(f"❌ Schema setup failed: {e}")
            raise
    
    async def upsert_entity(self, entity: EntityExtraction, project_id: str) -> str:
        """Insert or update entity in knowledge graph"""
        try:
            mutation = {
                'uid': '_:entity',
                'dgraph.type': 'Entity',
                'name': entity.name,
                'entity_type': entity.entity_type,
                'description': entity.description,
                'confidence_score': entity.confidence_score,
                'source_chunks': entity.source_chunks,
                'project_id': project_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
            
            # Add embeddings if available
            if entity.name_embedding:
                mutation['name_embedding'] = entity.name_embedding
            if entity.description_embedding:
                mutation['description_embedding'] = entity.description_embedding
            
            # Execute mutation
            txn = self.client.txn()
            try:
                response = await txn.mutate(set_obj=mutation)
                await txn.commit()
                
                # Extract assigned UID
                entity_uid = response.uids.get('entity')
                logger.info(f"✅ Entity '{entity.name}' upserted with UID: {entity_uid}")
                return entity_uid
                
            finally:
                await txn.discard()
                
        except Exception as e:
            logger.error(f"❌ Entity upsert failed for '{entity.name}': {e}")
            raise
    
    async def upsert_relationship(self, 
                                relationship: RelationshipExtraction, 
                                source_uid: str, 
                                target_uid: str,
                                project_id: str) -> str:
        """Insert or update relationship in knowledge graph"""
        try:
            mutation = {
                'uid': '_:relationship',
                'dgraph.type': 'Relationship',
                'name': f"{relationship.source_entity}_{relationship.relationship_type}_{relationship.target_entity}",
                'relationship_type': relationship.relationship_type,
                'description': relationship.description,
                'weight': relationship.weight,
                'confidence_score': relationship.confidence_score,
                'source_evidence': relationship.source_evidence,
                'source_entity': {'uid': source_uid},
                'target_entity': {'uid': target_uid},
                'project_id': project_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
            
            # Add embedding if available
            if relationship.description_embedding:
                mutation['description_embedding'] = relationship.description_embedding
            
            # Execute mutation
            txn = self.client.txn()
            try:
                response = await txn.mutate(set_obj=mutation)
                await txn.commit()
                
                relationship_uid = response.uids.get('relationship')
                logger.info(f"✅ Relationship '{relationship.relationship_type}' upserted with UID: {relationship_uid}")
                return relationship_uid
                
            finally:
                await txn.discard()
                
        except Exception as e:
            logger.error(f"❌ Relationship upsert failed: {e}")
            raise
    
    async def query_entities_by_project(self, project_id: str, limit: int = 100) -> List[Dict]:
        """Query entities for a specific project"""
        query = f"""
        {{
            entities(func: eq(project_id, "{project_id}")) @filter(type(Entity)) (first: {limit}) {{
                uid
                name
                entity_type
                description
                confidence_score
                name_embedding
                description_embedding
                created_at
            }}
        }}
        """
        
        try:
            txn = self.client.txn(read_only=True)
            try:
                response = await txn.query(query)
                result = json.loads(response.json)
                return result.get('entities', [])
            finally:
                await txn.discard()
                
        except Exception as e:
            logger.error(f"❌ Entity query failed for project {project_id}: {e}")
            return []
    
    async def query_relationships_by_entities(self, entity_uids: List[str]) -> List[Dict]:
        """Query relationships between specific entities"""
        uid_list = ', '.join([f'"{uid}"' for uid in entity_uids])
        
        query = f"""
        {{
            relationships(func: uid({uid_list})) {{
                ~source_entity {{
                    uid
                    name
                    relationship_type
                    description
                    weight
                    confidence_score
                    target_entity {{
                        uid
                        name
                        entity_type
                    }}
                }}
            }}
        }}
        """
        
        try:
            txn = self.client.txn(read_only=True)
            try:
                response = await txn.query(query)
                result = json.loads(response.json)
                return result.get('relationships', [])
            finally:
                await txn.discard()
                
        except Exception as e:
            logger.error(f"❌ Relationship query failed: {e}")
            return []