from flask import Blueprint, request, jsonify
import re
import json
from src.ai_engine import EnhancedChatbot

chatbot_bp = Blueprint('chatbot', __name__)

# Instância global do chatbot aprimorado
enhanced_chatbot = EnhancedChatbot()

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal para interação com o chatbot aprimorado"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Mensagem não fornecida'
            }), 400
        
        user_message = data['message']
        user_id = data.get('user_id', 'default')  # Suporte a múltiplos usuários
        
        # Processar mensagem com IA avançada
        response_data = enhanced_chatbot.process_message(user_message, user_id)
        
        return jsonify({
            'user_message': user_message,
            'bot_response': response_data['response'],
            'response_type': response_data['type'],
            'topic': response_data.get('topic', None),
            'confidence': response_data.get('confidence', 0.0),
            'user_id': user_id
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Erro interno: {str(e)}'
        }), 500

@chatbot_bp.route('/knowledge-base', methods=['GET'])
def get_knowledge_base():
    """Endpoint para obter a base de conhecimento"""
    return jsonify({
        'topics': list(enhanced_chatbot.knowledge_base.keys()),
        'knowledge_base': enhanced_chatbot.knowledge_base
    })

@chatbot_bp.route('/diagnostic/start', methods=['POST'])
def start_diagnostic():
    """Endpoint para iniciar um diagnóstico específico"""
    try:
        data = request.get_json()
        problem_type = data.get('problem_type')
        user_id = data.get('user_id', 'default')
        
        if not problem_type:
            return jsonify({
                'error': 'Tipo de problema não especificado'
            }), 400
        
        response = enhanced_chatbot.inference_engine.start_diagnostic(problem_type, user_id)
        
        return jsonify({
            'response': response,
            'problem_type': problem_type,
            'user_id': user_id
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Erro interno: {str(e)}'
        }), 500

@chatbot_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde"""
    return jsonify({
        'status': 'healthy',
        'service': 'Chatbot de Suporte Técnico com IA Avançada',
        'version': '2.0.0',
        'features': [
            'PLN Avançado',
            'Inferência Lógica',
            'Diagnóstico Interativo',
            'Múltiplos Usuários'
        ]
    })

@chatbot_bp.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint para estatísticas do sistema"""
    return jsonify({
        'total_topics': len(enhanced_chatbot.knowledge_base),
        'active_diagnostics': len(enhanced_chatbot.inference_engine.conversation_context),
        'inference_rules': len(enhanced_chatbot.inference_engine.rules),
        'synonyms_count': sum(len(synonyms) for synonyms in enhanced_chatbot.nlp_engine.synonyms.values())
    })

