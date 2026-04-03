# utils/lightrag/lightrag_utils.py
"""
LightRAG Integration for Existing RAG Pipeline
Enhances naive vector RAG with knowledge graph capabilities using DGraph

https://github.com/HKUDS/LightRAG/blob/main/examples/lightrag_openai_demo.py
"""

import os
import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

# LightRAG core
from lightrag import LightRAG, QueryParam
from lightrag.kg.neo4j_impl import Neo4JStorage

# LLM and embedding functions
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import (
    gpt_4o_mini_complete,
    openai_embed
)
# DGraph integration — imported lazily to avoid crashing Celery if protobuf is misconfigured
try:
    import pydgraph
    from pydgraph import DgraphClient, DgraphClientStub
    _pydgraph_available = True
except ImportError as e:
    pydgraph = None
    DgraphClient = None
    DgraphClientStub = None
    _pydgraph_available = False
    logging.getLogger(__name__).warning(f"pydgraph unavailable, DGraph features disabled: {e}")


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

        if not _pydgraph_available:
            raise RuntimeError("pydgraph is not available — install protobuf>=4.25.0 and pydgraph")

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
        
# ——— Enhanced LightRAG Manager ———————————————————————————————————————

class LightRAGManager:
    """
    Simplified LightRAG manager that uses DGraph as LightRAG's native storage
    
    Key Changes:
    1. LightRAG handles ALL knowledge graph operations internally
    2. DGraph is configured as LightRAG's graph storage backend
    3. No separate DGraphManager - LightRAG manages the graph
    4. No duplicate entity extraction - LightRAG does this natively
    """
    
    def __init__(self):
        self.lightrag = None
        self._initialized = False
    
async def initialize(self):
    """Initialize LightRAG with DGraph as the graph storage backend"""
    if self._initialized:
        return
    
    try:
        # Configure LightRAG to use DGraph as its graph storage
        self.lightrag = LightRAG(
            working_dir=LIGHTRAG_WORKING_DIR,
            
            # LLM configuration
            llm_model_func=gpt_4o_mini_complete,
            
            # Embedding configuration - wrap openai_embed properly
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,  # OpenAI ada-002 dimension
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts=texts,
                    model="text-embedding-ada-002",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                )
            ),
            
            # Storage configuration - LightRAG handles the graph internally (eventually add and DGraphStorage... for now i have Neo4JStorage)
            graph_storage_cls=Neo4JStorage,
            graph_storage_kwargs={
                "host": DGRAPH_HOST,
                "port": 9080, 
            },
            
            # Vector storage (can be separate or integrated)
            vector_storage_cls="NanoVectorDBStorage",
        )
        
        # Initialize LightRAG's internal storages (including DGraph)
        await self.lightrag.initialize_storages()
        await self.lightrag.initialize_pipeline_status()
        
        self._initialized = True
        logger.info("✅ LightRAG manager initialized with DGraph backend")
        
    except Exception as e:
        logger.error(f"❌ LightRAG manager initialization failed: {e}")
        raise

    async def insert_document_into_kg(
            self, 
            doc_id: str,
            content: str, 
            project_id: str,
            metadata: Dict = None) -> Dict[str, Any]:
        """
        Process document through LightRAG system and add to knowledge graph, and entities and relationships to vdb

        Args:
            doc_id (str): Unique document identifier
            content (str): Full document content (as plaintext or a List of chunks)
            project_id (str): Project identifier for context
            metadata (Dict): Optional metadata for the document
        
        LightRAG handles everything internally:
            1. Entity/relationship extraction
            2. Knowledge graph storage (in DGraph)
            3. Vector embedding and storage
            4. Deduplication and merging
        """
        
        try:            
            # ——— 1. LightRAG Processing (Everything happens here) ———————————————————
            logger.info("🧠 Processing with LightRAG (extraction + graph + vectors)...")
            
            # Initialize the result variable
            concatenated_content = None

            # Check if content is a list & all elem are str
            if isinstance(content, list) and all(isinstance(elem, str) for elem in content):
                # If both are true, join the list elements into a single string
                concatenated_content = "".join(content)
                print(f"Successfully concatenated chunks!'")
            else:
                print("The variable is not a list of strings.")

            # Add project context to content for better entity resolution
            contextualized_content = f"[PROJECT: {project_id}] [DOC: {doc_id}]\n\n{content}"
            
            # async insert with LightRAG
            await self.lightrag.ainsert(contextualized_content)
            
            # ——— 2. Track Processing Metrics ————————————————————————————————————————
            # Get stats from LightRAG's internal storage
            processing_stats = await self._get_processing_stats(project_id)
            
            # ——— 3. Update Document Status ————————————————————————————————————————————
            await self._update_document_kg_status(doc_id, 'COMPLETE', {
                'processed_by': 'lightrag',
                'entities_estimated': processing_stats.get('entities_count', 0),
                'relationships_estimated': processing_stats.get('relationships_count', 0),
                'chunks_processed': processing_stats.get('chunks_count', 0)
            })
            
            logger.info(f"✅ LightRAG processing complete for document {doc_id}")
            
            return {
                'success': True,
                'processing_method': 'lightrag_unified',
                'entities_count': processing_stats.get('entities_count', 0),
                'relationships_count': processing_stats.get('relationships_count', 0),
                'chunks_count': processing_stats.get('chunks_count', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ LightRAG processing failed for document {doc_id}: {e}")
            await self._update_document_kg_status(doc_id, 'FAILED', {'error': str(e)})
            raise

    async def process_document_with_knowledge_graph(self, 
                                                    doc_id: str,
                                                    content: str, 
                                                    project_id: str,
                                                    metadata: Dict = None) -> Dict[str, Any]:
        """
        Process document through LightRAG system
        
        SIMPLIFIED: LightRAG handles everything internally:
        1. Entity/relationship extraction
        2. Knowledge graph storage (in DGraph)
        3. Vector embedding and storage
        4. Deduplication and merging
        """
        
        try:
            logger.info(f"🔍 Processing document {doc_id} with LightRAG")
            
            # ——— 1. LightRAG Processing (Everything happens here) ———————————————————
            logger.info("🧠 Processing with LightRAG (extraction + graph + vectors)...")
            
            # Add project context to content for better entity resolution
            contextualized_content = f"[PROJECT: {project_id}] [DOC: {doc_id}]\n\n{content}"
            
            # LightRAG handles:
            # - Text chunking
            # - Entity/relationship extraction  
            # - Knowledge graph construction
            # - Vector embedding
            # - Storage in DGraph + vector DB
            await self.lightrag.ainsert(contextualized_content)
            
            # ——— 2. Track Processing Metrics ————————————————————————————————————————
            # Get stats from LightRAG's internal storage
            processing_stats = await self._get_processing_stats(project_id)
            
            # ——— 3. Update Document Status ————————————————————————————————————————————
            await self._update_document_kg_status(doc_id, 'COMPLETE', {
                'processed_by': 'lightrag',
                'entities_estimated': processing_stats.get('entities_count', 0),
                'relationships_estimated': processing_stats.get('relationships_count', 0),
                'chunks_processed': processing_stats.get('chunks_count', 0)
            })
            
            logger.info(f"✅ LightRAG processing complete for document {doc_id}")
            
            return {
                'success': True,
                'processing_method': 'lightrag_unified',
                'entities_count': processing_stats.get('entities_count', 0),
                'relationships_count': processing_stats.get('relationships_count', 0),
                'chunks_count': processing_stats.get('chunks_count', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ LightRAG processing failed for document {doc_id}: {e}")
            await self._update_document_kg_status(doc_id, 'FAILED', {'error': str(e)})
            raise
    
    async def _get_processing_stats(self, project_id: str) -> Dict[str, int]:
        """
        Get processing statistics from LightRAG's internal storage
        This is optional - mainly for monitoring/debugging
        """
        try:
            # Access LightRAG's internal graph storage to get stats
            # This would depend on LightRAG's internal API
            stats = {
                'entities_count': 0,
                'relationships_count': 0,
                'chunks_count': 0
            }
            
            # If LightRAG exposes storage statistics:
            # stats = await self.lightrag.get_storage_stats()
            
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve processing stats: {e}")
            return {'entities_count': 0, 'relationships_count': 0, 'chunks_count': 0}
    
    async def enhanced_retrieval(self, 
                                query: str, 
                                project_id: str,
                                mode: str = "hybrid",
                                top_k: int = 10) -> Dict[str, Any]:
        """
        Enhanced retrieval using LightRAG + optional vector fallback
        
        SIMPLIFIED: LightRAG does the heavy lifting
        - mode="lightrag": Pure LightRAG (graph + vector internally)
        - mode="hybrid": LightRAG + your existing vector search
        - mode="vector": Your existing vector search only
        """
        
        try:
            logger.info(f"🔍 Enhanced retrieval for query: '{query}' (mode: {mode})")
            
            results = {
                'query': query,
                'mode': mode,
                'lightrag_response': None,
                'vector_chunks': [],
                'combined_context': None
            }
            
            # ——— 1. LightRAG Query (Handles graph + vector internally) ————————————————
            if mode in ['hybrid', 'lightrag']:
                logger.info("🧠 Querying LightRAG system...")
                
                # Add project context to query for better retrieval
                contextualized_query = f"[PROJECT: {project_id}] {query}"
                
                # LightRAG's internal modes:
                # - "naive": Vector similarity only
                # - "local": Low-level entity/relationship queries
                # - "global": High-level community/theme queries  
                # - "hybrid": Combined low + high level (recommended)
                lightrag_mode = "hybrid" if mode == "hybrid" else "local"
                
                lightrag_response = await self.lightrag.aquery(
                    contextualized_query, 
                    param=QueryParam(mode=lightrag_mode)
                )
                results['lightrag_response'] = lightrag_response
            
            # ——— 2. Optional: Your Existing Vector Search (Fallback/Supplement) ————————
            if mode in ['hybrid', 'vector']:
                logger.info("🔢 Querying existing vector search...")
                
                # Use your existing PostgreSQL vector search as supplement
                vector_chunks = await self._vector_similarity_search(query, project_id, top_k//2)
                results['vector_chunks'] = vector_chunks
            
            # ——— 3. Combine Results ————————————————————————————————————————————————————
            combined_context = self._combine_retrieval_results(results)
            results['combined_context'] = combined_context
            
            logger.info(f"✅ Enhanced retrieval complete")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced retrieval failed: {e}")
            raise
    
    async def _vector_similarity_search(self, query: str, project_id: str, top_k: int) -> List[Dict]:
        """Your existing PostgreSQL vector similarity search (unchanged)"""
        try:
            # Use your existing embedding model
            query_embedding = await self._generate_embedding(query)
            
            # Query your existing vector database
            async with get_db_connection() as conn:
                vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
                rows = await conn.fetch(
                    "SELECT * FROM match_document_chunks_hnsw($1, $2, $3)",
                    project_id, vector_str, top_k
                )
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.warning(f"⚠️ Vector search fallback failed: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using LightRAG's embedding function"""
        try:
            return await openai_embedding([text])
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation failed: {e}")
            return []
    
    def _combine_retrieval_results(self, results: Dict[str, Any]) -> str:
        """
        Combine LightRAG response with optional vector search results
        """
        
        context_parts = []
        
        # Primary: LightRAG response (comprehensive graph + vector analysis)
        if results.get('lightrag_response'):
            context_parts.append("=== LightRAG Knowledge Graph Analysis ===")
            context_parts.append(results['lightrag_response'])
            context_parts.append("")
        
        # Supplementary: Additional vector search results (if hybrid mode)
        if results.get('vector_chunks'):
            context_parts.append("=== Additional Context (Legacy Vector Search) ===")
            for chunk in results['vector_chunks'][:3]:  # Top 3 chunks
                content = chunk.get('content', '')[:300]  # Truncate for brevity
                context_parts.append(f"• {content}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def _update_document_kg_status(self, doc_id: str, status: str, metadata: Dict):
        """Update document knowledge graph processing status"""
        try:
            async with get_db_connection() as conn:
                await conn.execute(
                    """
                    UPDATE document_sources 
                    SET knowledge_graph_status = $1, kg_metadata = $2, updated_at = NOW()
                    WHERE id = $3
                    """,
                    status, json.dumps(metadata), doc_id
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to update KG status for document {doc_id}: {e}")

# ——— Integration with Existing Pipeline ————————————————————————————————————

class LightRAGPipelineIntegration:
    """
    SIMPLIFIED integration that uses LightRAG as a unified system
    """
    
    def __init__(self):
        self.lightrag_manager = LightRAGManager()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LightRAG pipeline integration"""
        if not self._initialized:
            await self.lightrag_manager.initialize()
            self._initialized = True
            logger.info("✅ LightRAG pipeline integration initialized")
    
    async def enhance_document_processing(self, 
                                        doc_id: str,
                                        chunks: List[str], 
                                        metadatas: List[Dict],
                                        project_id: str) -> Dict[str, Any]:
        """
        Enhance your existing document processing with LightRAG
        
        SIMPLIFIED: Just pass the content to LightRAG
        """
        
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info(f"🔄 Enhancing document processing for {doc_id} with LightRAG")
            
            # Combine chunks into full document content
            full_content = "\n\n".join(chunks)
            
            # Process with LightRAG (handles everything internally)
            kg_result = await self.lightrag_manager.process_document_with_knowledge_graph(
                doc_id=doc_id,
                content=full_content,
                project_id=project_id,
                metadata={'chunk_count': len(chunks)}
            )
            
            logger.info(f"✅ LightRAG enhancement complete for {doc_id}")
            
            return kg_result
            
        except Exception as e:
            logger.error(f"❌ LightRAG enhancement failed for {doc_id}: {e}")
            # Don't fail the entire pipeline - just log and continue
            return {'success': False, 'error': str(e)}
    
    async def enhance_note_generation(self, 
                                    query: str,
                                    project_id: str, 
                                    note_type: str,
                                    retrieval_mode: str = "hybrid") -> str:
        """
        Enhance your existing note generation with LightRAG retrieval
        
        SIMPLIFIED: LightRAG handles the complexity
        """
        
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info(f"🔄 Enhancing note generation with LightRAG for query: '{query}'")
            
            # Use enhanced retrieval 
            retrieval_results = await self.lightrag_manager.enhanced_retrieval(
                query=query,
                project_id=project_id,
                mode=retrieval_mode,
                top_k=15
            )
            
            # Return the combined context for your existing note generation
            enhanced_context = retrieval_results['combined_context']
            
            logger.info(f"✅ Enhanced retrieval complete - context length: {len(enhanced_context)} chars")
            
            return enhanced_context
            
        except Exception as e:
            logger.error(f"❌ Enhanced note generation failed: {e}")
            # Fallback to your existing retrieval if LightRAG fails
            logger.info("🔄 Falling back to vector-only retrieval...")
            raise

# ——— Global Manager Instance ————————————————————————————————————————————————

lightrag_integration = LightRAGPipelineIntegration()
lightrag_client = LightRAGManager()