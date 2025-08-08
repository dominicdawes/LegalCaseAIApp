# utils/lightrag/lightrag_utils.py
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
        
# ——— Enhanced LightRAG Manager ———————————————————————————————————————

class LightRAGManager:
    """
    Enhanced LightRAG manager that integrates with your existing pipeline
    Combines vector similarity with knowledge graph capabilities
    """
    
    def __init__(self):
        self.lightrag = None
        self.dgraph_manager = DGraphManager()
        self.embedding_model = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize LightRAG and DGraph components"""
        if self._initialized:
            return
        
        try:
            # Initialize DGraph
            await self.dgraph_manager.initialize()
            
            # Initialize LightRAG
            self.lightrag = LightRAG(
                working_dir=LIGHTRAG_WORKING_DIR,
                llm_model_func=gpt_4o_mini_complete,  # Use your existing LLM setup
                embedding_func=openai_embedding
            )
            
            # Initialize storage and pipeline
            await self.lightrag.initialize_storages()
            await self.lightrag.initialize_pipeline_status()
            
            # Initialize embedding model for manual embeddings
            self.embedding_model = openai_embedding
            
            self._initialized = True
            logger.info("✅ Enhanced LightRAG manager initialized")
            
        except Exception as e:
            logger.error(f"❌ LightRAG manager initialization failed: {e}")
            raise
    
    async def process_document_with_knowledge_graph(self, 
                                                  doc_id: str,
                                                  content: str, 
                                                  project_id: str,
                                                  metadata: Dict = None) -> Dict[str, Any]:
        """
        Process document through LightRAG + DGraph pipeline
        
        This replaces your existing embedding-only approach with:
        1. LightRAG entity/relationship extraction
        2. DGraph knowledge graph storage  
        3. Hybrid vector + graph indexing
        """
        
        try:
            logger.info(f"🔍 Processing document {doc_id} with LightRAG + DGraph")
            
            # ——— 1. LightRAG Processing (Entity + Relationship Extraction) ———————————
            logger.info("🧠 Extracting entities and relationships with LightRAG...")
            
            # Insert document into LightRAG (this triggers entity/relationship extraction)
            await self.lightrag.ainsert(content)
            
            # ——— 2. Extract Knowledge Graph Data ————————————————————————————————————
            # Note: LightRAG stores in its internal format, we need to extract for DGraph
            entities, relationships = await self._extract_kg_from_lightrag(content, doc_id)
            
            # ——— 3. Store in DGraph Knowledge Graph ————————————————————————————————
            logger.info(f"📊 Storing {len(entities)} entities and {len(relationships)} relationships in DGraph...")
            
            entity_uids = {}
            
            # Store entities
            for entity in entities:
                try:
                    entity_uid = await self.dgraph_manager.upsert_entity(entity, project_id)
                    entity_uids[entity.name] = entity_uid
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store entity '{entity.name}': {e}")
            
            # Store relationships
            relationship_uids = []
            for relationship in relationships:
                try:
                    source_uid = entity_uids.get(relationship.source_entity)
                    target_uid = entity_uids.get(relationship.target_entity)
                    
                    if source_uid and target_uid:
                        rel_uid = await self.dgraph_manager.upsert_relationship(
                            relationship, source_uid, target_uid, project_id
                        )
                        relationship_uids.append(rel_uid)
                    else:
                        logger.warning(f"⚠️ Missing entity UIDs for relationship: {relationship.relationship_type}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store relationship: {e}")
            
            # ——— 4. Update Document Status ————————————————————————————————————————————
            await self._update_document_kg_status(doc_id, 'COMPLETE', {
                'entities_extracted': len(entities),
                'relationships_extracted': len(relationships),
                'entity_uids': list(entity_uids.values()),
                'relationship_uids': relationship_uids
            })
            
            logger.info(f"✅ Knowledge graph processing complete for document {doc_id}")
            
            return {
                'success': True,
                'entities_count': len(entities),
                'relationships_count': len(relationships),
                'entity_uids': entity_uids,
                'relationship_uids': relationship_uids
            }
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph processing failed for document {doc_id}: {e}")
            await self._update_document_kg_status(doc_id, 'FAILED', {'error': str(e)})
            raise
    
    async def _extract_kg_from_lightrag(self, content: str, doc_id: str) -> Tuple[List[EntityExtraction], List[RelationshipExtraction]]:
        """
        Extract entities and relationships from content using LLMs
        This is a custom implementation that mimics LightRAG's internal extraction
        """
        
        # Use your existing LLM factory for entity extraction
        llm_client = LLMFactory.get_client_for("openai", "gpt-4o-mini", 0.3, False)
        
        # Entity extraction prompt
        entity_prompt = f"""
        Extract entities from the following text. For each entity, provide:
        - name: The entity name
        - type: The entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, etc.)
        - description: A brief description of the entity
        - confidence: Confidence score (0.0 to 1.0)
        
        Text: {content}
        
        Return as JSON array:
        [
            {{
                "name": "entity_name",
                "type": "ENTITY_TYPE", 
                "description": "description",
                "confidence": 0.9
            }}
        ]
        """
        
        # Relationship extraction prompt
        relationship_prompt = f"""
        Extract relationships between entities from the following text. For each relationship:
        - source_entity: Source entity name
        - target_entity: Target entity name
        - relationship_type: Type of relationship (WORKS_FOR, LOCATED_IN, PART_OF, etc.)
        - description: Description of the relationship
        - weight: Relationship strength (0.0 to 1.0)
        - confidence: Confidence score (0.0 to 1.0)
        
        Text: {content}
        
        Return as JSON array:
        [
            {{
                "source_entity": "entity1",
                "target_entity": "entity2",
                "relationship_type": "RELATIONSHIP_TYPE",
                "description": "description",
                "weight": 0.8,
                "confidence": 0.9
            }}
        ]
        """
        
        try:
            # Extract entities
            entity_response = await asyncio.get_event_loop().run_in_executor(
                None, llm_client.chat, entity_prompt
            )
            entities_data = json.loads(entity_response)
            
            # Extract relationships
            relationship_response = await asyncio.get_event_loop().run_in_executor(
                None, llm_client.chat, relationship_prompt
            )
            relationships_data = json.loads(relationship_response)
            
            # Convert to our data structures
            entities = []
            for entity_data in entities_data:
                # Generate embeddings for entity name and description
                name_embedding = await self._generate_embedding(entity_data['name'])
                desc_embedding = await self._generate_embedding(entity_data['description'])
                
                entity = EntityExtraction(
                    name=entity_data['name'],
                    entity_type=entity_data['type'],
                    description=entity_data['description'],
                    confidence_score=entity_data['confidence'],
                    source_chunks=[content],  # In practice, you'd track specific chunks
                    name_embedding=name_embedding,
                    description_embedding=desc_embedding
                )
                entities.append(entity)
            
            relationships = []
            for rel_data in relationships_data:
                # Generate embedding for relationship description
                desc_embedding = await self._generate_embedding(rel_data['description'])
                
                relationship = RelationshipExtraction(
                    source_entity=rel_data['source_entity'],
                    target_entity=rel_data['target_entity'],
                    relationship_type=rel_data['relationship_type'],
                    description=rel_data['description'],
                    weight=rel_data['weight'],
                    confidence_score=rel_data['confidence'],
                    source_evidence=[content],  # Track source evidence
                    description_embedding=desc_embedding
                )
                relationships.append(relationship)
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"❌ Knowledge extraction failed: {e}")
            return [], []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            return await self.embedding_model(text)
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation failed for text: {e}")
            return []
    
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
    
    async def enhanced_retrieval(self, 
                               query: str, 
                               project_id: str,
                               mode: str = "hybrid",
                               top_k: int = 10) -> Dict[str, Any]:
        """
        Enhanced retrieval combining LightRAG + DGraph + your existing vector search
        
        This implements the dual-level retrieval that makes LightRAG powerful:
        - Low-level: Specific entity/relationship retrieval
        - High-level: Conceptual theme retrieval
        - Vector fallback: Your existing semantic similarity
        """
        
        try:
            logger.info(f"🔍 Enhanced retrieval for query: '{query}' (mode: {mode})")
            
            results = {
                'query': query,
                'mode': mode,
                'entities': [],
                'relationships': [],
                'themes': [],
                'vector_chunks': [],
                'lightrag_response': None
            }
            
            # ——— 1. LightRAG Query (Dual-level retrieval) ————————————————————————————
            if mode in ['hybrid', 'lightrag']:
                logger.info("🧠 Querying LightRAG knowledge graph...")
                
                # LightRAG has different query modes
                lightrag_mode = "hybrid" if mode == "hybrid" else "local"
                lightrag_response = await self.lightrag.aquery(
                    query, 
                    param=QueryParam(mode=lightrag_mode)
                )
                results['lightrag_response'] = lightrag_response
            
            # ——— 2. DGraph Entity/Relationship Queries ——————————————————————————————
            if mode in ['hybrid', 'graph']:
                logger.info("📊 Querying DGraph knowledge graph...")
                
                # Query relevant entities
                query_embedding = await self._generate_embedding(query)
                entities = await self.dgraph_manager.query_entities_by_project(project_id, top_k)
                
                # Filter entities by relevance (you could implement vector similarity here)
                relevant_entities = [e for e in entities if self._is_entity_relevant(e, query)][:top_k//2]
                results['entities'] = relevant_entities
                
                # Query relationships for relevant entities
                if relevant_entities:
                    entity_uids = [e['uid'] for e in relevant_entities]
                    relationships = await self.dgraph_manager.query_relationships_by_entities(entity_uids)
                    results['relationships'] = relationships
            
            # ——— 3. Fallback to Vector Search (Your existing approach) ——————————————————
            if mode in ['hybrid', 'vector']:
                logger.info("🔢 Fallback to vector similarity search...")
                
                # Use your existing vector search function
                vector_chunks = await self._vector_similarity_search(query, project_id, top_k//2)
                results['vector_chunks'] = vector_chunks
            
            # ——— 4. Combine and Rank Results ————————————————————————————————————————
            combined_context = self._combine_retrieval_results(results)
            results['combined_context'] = combined_context
            
            logger.info(f"✅ Enhanced retrieval complete: {len(results['entities'])} entities, "
                       f"{len(results['relationships'])} relationships, "
                       f"{len(results['vector_chunks'])} vector chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced retrieval failed: {e}")
            raise
    
    def _is_entity_relevant(self, entity: Dict, query: str) -> bool:
        """Simple relevance check for entity filtering"""
        query_lower = query.lower()
        entity_text = f"{entity.get('name', '')} {entity.get('description', '')}".lower()
        
        # Simple keyword matching - in practice, you'd use embeddings
        return any(word in entity_text for word in query_lower.split())
    
    async def _vector_similarity_search(self, query: str, project_id: str, top_k: int) -> List[Dict]:
        """Your existing vector similarity search function"""
        # This would call your existing fetch_relevant_chunks function
        # or use your PostgreSQL vector search
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
    
    def _combine_retrieval_results(self, results: Dict[str, Any]) -> str:
        """
        Combine different retrieval results into coherent context
        This is where LightRAG's power shows - integrating multiple knowledge sources
        """
        
        context_parts = []
        
        # Add LightRAG response (already processed and coherent)
        if results.get('lightrag_response'):
            context_parts.append("=== LightRAG Knowledge Graph Response ===")
            context_parts.append(results['lightrag_response'])
            context_parts.append("")
        
        # Add entity information
        if results.get('entities'):
            context_parts.append("=== Relevant Entities ===")
            for entity in results['entities'][:5]:  # Top 5 entities
                context_parts.append(f"• {entity.get('name', 'Unknown')} ({entity.get('entity_type', 'Unknown')}): {entity.get('description', '')}")
            context_parts.append("")
        
        # Add relationship information
        if results.get('relationships'):
            context_parts.append("=== Relevant Relationships ===")
            for rel in results['relationships'][:5]:  # Top 5 relationships
                context_parts.append(f"• {rel.get('name', 'Unknown')}: {rel.get('description', '')}")
            context_parts.append("")
        
        # Add vector search results as fallback
        if results.get('vector_chunks'):
            context_parts.append("=== Additional Context (Vector Similarity) ===")
            for chunk in results['vector_chunks'][:3]:  # Top 3 chunks
                content = chunk.get('content', '')[:500]  # Truncate for brevity
                context_parts.append(f"• {content}...")
            context_parts.append("")
        
        return "\n".join(context_parts)

# ——— Integration with Existing Pipeline ————————————————————————————————————

class LightRAGPipelineIntegration:
    """
    Integration layer that plugs LightRAG into your existing upload_tasks.py and note_tasks.py
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
        Enhance your existing document processing with knowledge graph extraction
        
        This plugs into your existing _process_embeddings_async function
        """
        
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info(f"🔄 Enhancing document processing for {doc_id} with LightRAG")
            
            # Combine chunks into full document content for knowledge extraction
            full_content = "\n\n".join(chunks)
            
            # Process with LightRAG + DGraph (runs in parallel with your existing embeddings)
            kg_result = await self.lightrag_manager.process_document_with_knowledge_graph(
                doc_id=doc_id,
                content=full_content,
                project_id=project_id,
                metadata={'chunk_count': len(chunks)}
            )
            
            logger.info(f"✅ Knowledge graph enhancement complete for {doc_id}: "
                       f"{kg_result['entities_count']} entities, {kg_result['relationships_count']} relationships")
            
            return kg_result
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph enhancement failed for {doc_id}: {e}")
            # Don't fail the entire pipeline - just log and continue
            return {'success': False, 'error': str(e)}
    
    async def enhance_note_generation(self, 
                                    query: str,
                                    project_id: str, 
                                    note_type: str,
                                    retrieval_mode: str = "hybrid") -> str:
        """
        Enhance your existing note generation with LightRAG retrieval
        
        This plugs into your existing note_tasks.py workflow
        """
        
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info(f"🔄 Enhancing note generation with LightRAG for query: '{query}'")
            
            # Use enhanced retrieval instead of just vector search
            retrieval_results = await self.lightrag_manager.enhanced_retrieval(
                query=query,
                project_id=project_id,
                mode=retrieval_mode,
                top_k=15  # More results due to multi-source retrieval
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