from typing import List, Optional
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Importar para reconstrução
from langchain_core.chat_history import BaseChatMessageHistory
import datetime # <<< NOVO
import pytz # <<< NOVO
import json # <<< NOVO: Para desserializar o JSON da mensagem
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    # Fallback para psycopg 3.x
    import psycopg as psycopg2
    from psycopg import sql
from config.settings import settings


class LimitedPostgresChatMessageHistory(BaseChatMessageHistory):
    """PostgreSQL chat message history that stores all messages but limits agent context to recent messages."""
    
    # Fuso horário para formatação (padrão de time_tool)
    TIMEZONE = "America/Sao_Paulo" # <<< NOVO CONSTANTE
    
    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "message_store",
        max_messages: int = 20,
        **kwargs
    ):
        """
        Initialize limited PostgreSQL chat history.
        
        Args:
            session_id: Unique identifier for the chat session
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store messages
            max_messages: Maximum number of recent messages to return to the agent (default: 20)
        """
        self.session_id = session_id
        self.connection_string = connection_string
        self.table_name = table_name
        self.max_messages = max_messages
        
        # Initialize the base PostgreSQL history (stores all messages)
        self._postgres_history = PostgresChatMessageHistory(
            session_id=session_id,
            connection_string=connection_string,
            table_name=table_name,
            **kwargs
        )
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get optimized messages for the agent context."""
        return self.get_optimized_context()
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the database (all messages are stored)."""
        self._postgres_history.add_message(message)
    
    def clear(self) -> None:
        """Clear all messages for this session."""
        self._postgres_history.clear()
    
    def _enforce_message_limit(self) -> None:
        """Keep only the most recent max_messages messages."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Get message IDs ordered by ID (oldest first)
                    cursor.execute(f"""
                        SELECT id FROM {self.table_name}
                        WHERE session_id = %s
                        ORDER BY id ASC
                    """, (self.session_id,))
                    
                    message_ids = cursor.fetchall()
                    
                    # If we have more messages than the limit, delete the oldest ones
                    if len(message_ids) > self.max_messages:
                        messages_to_delete = len(message_ids) - self.max_messages
                        ids_to_delete = [msg[0] for msg in message_ids[:messages_to_delete]]
                        
                        # Usando psycopg2.sql para construção segura (se psycopg2 for usado)
                        try:
                            delete_query = psycopg2.sql.SQL(
                                "DELETE FROM {} WHERE id = ANY(%s)"
                            ).format(psycopg2.sql.Identifier(self.table_name))
                            cursor.execute(delete_query, (ids_to_delete,))
                        except AttributeError:
                             # Fallback para string formatada
                            cursor.execute(f"""
                                DELETE FROM {self.table_name}
                                WHERE id = ANY(%s)
                            """, (ids_to_delete,))
                        
                        conn.commit()
                        
                        print(f"Limited messages for session {self.session_id}: "
                              f"deleted {messages_to_delete} oldest messages, "
                              f"keeping {self.max_messages} most recent")
                              
        except Exception as e:
            print(f"Error enforcing message limit: {e}")
    
    def get_message_count(self) -> int:
        """Get the current number of messages for this session."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {self.table_name}
                        WHERE session_id = %s
                    """, (self.session_id,))
                    
                    return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting message count: {e}")
            return 0
    
    def get_session_info(self) -> dict:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "message_count": self.get_message_count(),
            "max_messages": self.max_messages,
            "table_name": self.table_name
        }
    
    def should_clear_context(self, recent_messages: List[BaseMessage]) -> bool:
        """
        Determine if context should be cleared based on recent messages.
        Returns True if agent is struggling to identify products.
        """
        if len(recent_messages) < 3:
            return False
            
        # Check if last few messages show agent confusion
        confusion_patterns = [
            "não identifiquei",
            "não consegui identificar",
            "informar o nome principal",
            "desculpe, não",
            "pode informar"
        ]
        
        recent_text = " ".join([msg.content.lower() for msg in recent_messages[-3:]])
        
        confusion_count = sum(1 for pattern in confusion_patterns if pattern in recent_text)
        
        # If 2+ confusion patterns in last 3 messages, suggest clearing
        return confusion_count >= 2
    
    
    def _fetch_messages_with_timestamp(self) -> List[BaseMessage]:
        """
        Recupera todas as mensagens COM a data de criação e injeta a data/hora no conteúdo.
        """
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Consulta SQL para recuperar a mensagem (JSONB) e a data de criação
                    query = f"""
                        SELECT message, created_at
                        FROM {self.table_name}
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                    """
                    cursor.execute(query, (self.session_id,))
                    
                    raw_messages = cursor.fetchall()

                    # Reconstruir o objeto BaseMessage e injetar o timestamp
                    processed_messages: List[BaseMessage] = []
                    tz = pytz.timezone(self.TIMEZONE)
                    
                    for raw_message_data in raw_messages:
                        message_json = raw_message_data[0]
                        created_at = raw_message_data[1] # datetime.datetime object
                        
                        # O campo 'message' no DB é um JSONB que armazena a representação da Langchain
                        # Desserializar para um dicionário Python
                        if isinstance(message_json, str):
                            message_dict = json.loads(message_json)
                        elif isinstance(message_json, dict):
                            message_dict = message_json
                        else:
                            continue # Pular dados inválidos
                            
                        message_type = message_dict.get("type", "human")
                        content = message_dict.get("content", "")
                        additional_kwargs = message_dict.get("additional_kwargs", {})
                        
                        # Formatar a hora
                        dt_localized = created_at.astimezone(tz)
                        formatted_time = dt_localized.strftime("%d/%m/%Y %H:%M:%S (%Z)")
                        
                        # Injetar o timestamp no conteúdo
                        new_content = f"[CONTEXTO_MEMORIA_ANTIGA: {formatted_time}] {content}"
                        
                        # Recriar o objeto BaseMessage com o conteúdo modificado
                        if message_type == "ai":
                            msg = AIMessage(content=new_content, additional_kwargs=additional_kwargs)
                        else:
                            # Tratar "human" e outros como HumanMessage
                            msg = HumanMessage(content=new_content, additional_kwargs=additional_kwargs)

                        processed_messages.append(msg)
                        
                    return processed_messages
                    
        except Exception as e:
            print(f"Error fetching messages with timestamp: {e}")
            # Em caso de falha, retorna o histórico padrão (sem timestamp injetado)
            return self._postgres_history.messages

    
    def get_optimized_context(self) -> List[BaseMessage]:
        """
        Get optimized context for product identification.
        Focuses on recent product-related messages.
        """
        # 1. Busca todas as mensagens COM o timestamp injetado
        all_messages = self._fetch_messages_with_timestamp()
        
        if len(all_messages) <= self.max_messages:
            return all_messages
        
        # 2. Obtém mensagens recentes (com timestamp injetado)
        # O histórico já está ordenado por created_at ASC
        recent_messages = all_messages[-self.max_messages:]
        
        # 3. Checa se deve limpar o contexto
        # A checagem de confusão deve ser feita no conteúdo JÁ INJETADO
        if self.should_clear_context(recent_messages):
            print(f"🔄 Detectada confusão do agente. Recomendação: limpar contexto para {self.session_id}")
            # Retorna apenas as 3 últimas mensagens (que já contêm o timestamp)
            return recent_messages[-3:]
        
        # 4. Retorna as N mensagens mais recentes (que já contêm o timestamp)
        return recent_messages
