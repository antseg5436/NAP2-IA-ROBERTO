"""
Módulo de IA avançada para chatbot de suporte técnico
Inclui PLN aprimorada, inferência lógica e fluxos interativos para todas as categorias
"""

import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

class AdvancedNLPEngine:
    """Engine de PLN avançada com processamento de texto melhorado"""
    
    def __init__(self):
        # Stopwords em português
        self.stopwords = {
            'a', 'o', 'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'por', 'que', 'se', 'na', 'no',
            'ao', 'as', 'os', 'das', 'dos', 'mas', 'ou', 'como', 'mais', 'muito', 'ja', 'nao', 'sao', 'tem',
            'foi', 'ser', 'ter', 'seu', 'sua', 'seus', 'suas', 'ele', 'ela', 'eles', 'elas', 'isso', 'isto',
            'aqui', 'ali', 'la', 'onde', 'quando', 'porque', 'como', 'qual', 'quais', 'quem', 'quanto', 'quantos'
        }
        
        # Sinônimos expandidos para melhor reconhecimento
        self.synonyms = {
            'wifi': ['wifi', 'wi-fi', 'internet', 'rede', 'conexao', 'conectar', 'wireless', 'sem fio', 'roteador', 
                    'modem', 'router', 'sinal', 'banda larga', 'broadband', 'net', 'web', 'online'],
            'internet': ['internet', 'net', 'web', 'online', 'conexao', 'conectar', 'rede', 'banda larga'],
            'senha': ['senha', 'password', 'pass', 'credencial', 'login', 'acesso', 'autenticacao', 'usuario',
                     'logon', 'entrar', 'acessar', 'logar', 'autenticar'],
            'impressora': ['impressora', 'printer', 'imprimir', 'impressao', 'papel', 'tinta', 'cartucho', 'toner',
                          'scanner', 'digitalizar', 'escanear', 'multifuncional', 'hp', 'canon', 'epson'],
            'email': ['email', 'e-mail', 'correio', 'mail', 'outlook', 'gmail', 'yahoo', 'hotmail', 'mensagem',
                     'enviar', 'receber', 'caixa de entrada', 'spam', 'lixo eletronico'],
            'problema': ['problema', 'erro', 'falha', 'bug', 'defeito', 'nao funciona', 'travando', 'lento',
                        'parou', 'quebrado', 'ruim', 'mal', 'dificuldade', 'complicacao'],
            'configurar': ['configurar', 'config', 'setup', 'instalar', 'ajustar', 'definir', 'estabelecer'],
            'ajuda': ['ajuda', 'help', 'socorro', 'auxilio', 'suporte', 'assistencia', 'apoio'],
            'rapido': ['rapido', 'fast', 'veloz', 'agil', 'ligeiro', 'presto'],
            'lento': ['lento', 'slow', 'devagar', 'demorado', 'tardio', 'moroso']
        }
        
        # Frases completas para cada categoria
        self.phrase_patterns = {
            'wifi': [
                'como configurar wifi', 'nao consigo conectar na internet', 'wifi nao funciona',
                'problema com internet', 'rede sem fio', 'configurar roteador', 'senha do wifi',
                'internet lenta', 'sinal fraco', 'nao conecta no wifi', 'wifi desconectando',
                'minha internet nao funciona', 'roteador com problemas', 'modem nao liga',
                'sem acesso a internet', 'conexao instavel', 'wifi cai toda hora'
            ],
            'senha': [
                'esqueci minha senha', 'como resetar senha', 'recuperar senha', 'nao lembro a senha',
                'perdi minha senha', 'redefinir password', 'problema de login', 'nao consigo entrar',
                'senha incorreta', 'bloqueado por senha', 'alterar senha', 'trocar senha',
                'senha nao funciona', 'conta bloqueada', 'acesso negado'
            ],
            'impressora': [
                'impressora nao funciona', 'nao consigo imprimir', 'problema na impressora',
                'impressora offline', 'papel atolado', 'tinta acabou', 'cartucho vazio',
                'impressora nao conecta', 'erro de impressao', 'impressora lenta',
                'nao reconhece impressora', 'driver da impressora', 'instalar impressora'
            ],
            'email': [
                'como configurar email', 'problema no email', 'nao recebo emails',
                'email nao funciona', 'configurar outlook', 'gmail nao abre',
                'problema com correio', 'nao consigo enviar email', 'email travando',
                'configurar conta de email', 'servidor de email', 'smtp pop3'
            ]
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """Pré-processa o texto removendo pontuação e normalizando"""
        # Converter para minúsculas
        text = text.lower()
        
        # Remover acentos básicos
        text = text.replace('ã', 'a').replace('á', 'a').replace('à', 'a').replace('â', 'a')
        text = text.replace('é', 'e').replace('ê', 'e').replace('í', 'i').replace('ó', 'o')
        text = text.replace('ô', 'o').replace('õ', 'o').replace('ú', 'u').replace('ü', 'u')
        text = text.replace('ç', 'c').replace('ñ', 'n')
        
        # Remover pontuação e caracteres especiais
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenizar
        tokens = text.split()
        
        # Remover stopwords
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        
        return tokens
    
    def calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calcula similaridade entre dois conjuntos de tokens com ponderação"""
        if not tokens1 or not tokens2:
            return 0.0
        
        # Converter para conjuntos para operações de interseção
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Correspondência exata
        exact_matches = len(set1.intersection(set2))
        
        # Correspondência por sinônimos
        synonym_matches = 0
        for token1 in set1:
            for token2 in set2:
                if self._are_synonyms(token1, token2):
                    synonym_matches += 1
                    break
        
        # Correspondência por similaridade de string (para erros de digitação)
        fuzzy_matches = 0
        for token1 in set1:
            for token2 in set2:
                if self._fuzzy_match(token1, token2):
                    fuzzy_matches += 0.5  # Peso menor para correspondências fuzzy
                    break
        
        # Calcular score total
        total_matches = exact_matches + synonym_matches + fuzzy_matches
        max_tokens = max(len(set1), len(set2))
        
        # Ponderação: dar mais peso para correspondências exatas e sinônimos
        similarity = (exact_matches * 2.0 + synonym_matches * 1.5 + fuzzy_matches) / (max_tokens * 2.0)
        
        # Bonus para N-grams (bigrams)
        bigram_bonus = self._calculate_bigram_similarity(tokens1, tokens2)
        
        return min(similarity + bigram_bonus, 3.0)  # Máximo de 3.0
    
    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """Verifica se duas palavras são sinônimos"""
        for category, synonyms in self.synonyms.items():
            if word1 in synonyms and word2 in synonyms:
                return True
        return False
    
    def _fuzzy_match(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """Verifica correspondência fuzzy para erros de digitação"""
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        similarity = SequenceMatcher(None, word1, word2).ratio()
        return similarity >= threshold
    
    def _calculate_bigram_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calcula similaridade baseada em bigrams"""
        if len(tokens1) < 2 or len(tokens2) < 2:
            return 0.0
        
        bigrams1 = set(zip(tokens1[:-1], tokens1[1:]))
        bigrams2 = set(zip(tokens2[:-1], tokens2[1:]))
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return (intersection / union) * 0.5  # Bonus de até 0.5
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classifica a intenção do texto com score de confiança"""
        tokens = self.preprocess_text(text)
        text_lower = text.lower()
        
        # Verificar correspondência com frases completas primeiro
        best_intent = None
        best_score = 0.0
        
        for intent, phrases in self.phrase_patterns.items():
            for phrase in phrases:
                phrase_tokens = self.preprocess_text(phrase)
                similarity = self.calculate_similarity(tokens, phrase_tokens)
                
                # Bonus para correspondência de frase completa
                if phrase in text_lower:
                    similarity += 1.0
                
                if similarity > best_score:
                    best_score = similarity
                    best_intent = intent
        
        # Se não encontrou correspondência boa com frases, tentar com sinônimos
        if best_score < 1.5:
            for intent, synonyms in self.synonyms.items():
                if intent in ['wifi', 'senha', 'impressora', 'email']:
                    synonym_tokens = synonyms
                    similarity = self.calculate_similarity(tokens, synonym_tokens)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = intent
        
        return (best_intent, best_score) if best_intent else ("unknown", 0.0)


class LogicalInferenceEngine:
    """Engine de inferência lógica com fluxos interativos para todos os diagnósticos"""
    
    def __init__(self):
        self.conversation_context = {}
        
        # Fluxos interativos para todas as categorias
        self.interactive_flows = {
            'password_recovery': {
                'steps': {
                    'start': {
                        'message': "Olá! Vejo que você selecionou a opção 'Senha'. Quer recuperar o acesso à sua conta?",
                        'expected_responses': {
                            'sim': 'access_login',
                            'yes': 'access_login',
                            'claro': 'access_login',
                            'quero': 'access_login',
                            'preciso': 'access_login',
                            'nao': 'different_problem',
                            'não': 'different_problem',
                            'no': 'different_problem'
                        },
                        'fallback_message': "Não entendi sua resposta. Você quer recuperar o acesso à sua conta? Responda 'sim' ou 'não'."
                    },
                    'access_login': {
                        'message': "Certo, vou te ajudar 👍\nPrimeiro passo: acesse a página de login do sistema.\nConseguiu chegar lá?",
                        'expected_responses': {
                            'sim': 'click_forgot',
                            'yes': 'click_forgot',
                            'consegui': 'click_forgot',
                            'ok': 'click_forgot',
                            'nao': 'help_find_login',
                            'não': 'help_find_login',
                            'no': 'help_find_login'
                        },
                        'fallback_message': "Você conseguiu acessar a página de login? Responda 'sim' se conseguiu ou 'não' se precisa de ajuda para encontrá-la."
                    },
                    'help_find_login': {
                        'message': "Sem problemas! Para encontrar a página de login:\n1. Abra seu navegador\n2. Digite o endereço do site/sistema\n3. Procure por 'Login', 'Entrar' ou 'Acesso'\n\nConseguiu encontrar agora?",
                        'expected_responses': {
                            'sim': 'click_forgot',
                            'yes': 'click_forgot',
                            'consegui': 'click_forgot',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu encontrar a página de login agora? Responda 'sim' ou 'não'."
                    },
                    'click_forgot': {
                        'message': "Ótimo! Agora clique em 'Esqueci minha senha'.\nEstá vendo essa opção?",
                        'expected_responses': {
                            'sim': 'enter_email',
                            'yes': 'enter_email',
                            'vejo': 'enter_email',
                            'cliquei': 'enter_email',
                            'ja cliquei': 'enter_email',
                            'nao': 'help_find_forgot',
                            'não': 'help_find_forgot',
                            'no': 'help_find_forgot'
                        },
                        'fallback_message': "Você está vendo a opção 'Esqueci minha senha'? Responda 'sim' se vê ou 'não' se não encontra."
                    },
                    'help_find_forgot': {
                        'message': "A opção pode estar com nomes como:\n• 'Esqueci minha senha'\n• 'Recuperar senha'\n• 'Forgot password'\n• 'Reset password'\n\nGeralmente fica abaixo dos campos de login. Encontrou?",
                        'expected_responses': {
                            'sim': 'enter_email',
                            'yes': 'enter_email',
                            'encontrei': 'enter_email',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu encontrar a opção para recuperar senha? Responda 'sim' ou 'não'."
                    },
                    'enter_email': {
                        'message': "Perfeito 👌\nAgora digite o seu e-mail cadastrado e clique em enviar.",
                        'expected_responses': {
                            'sim': 'check_email',
                            'yes': 'check_email',
                            'pronto': 'check_email',
                            'feito': 'check_email',
                            'ja fiz': 'check_email',
                            'digitei': 'check_email',
                            'enviei': 'check_email'
                        },
                        'fallback_message': "Você já digitou seu e-mail e clicou em enviar? Responda quando tiver feito isso."
                    },
                    'check_email': {
                        'message': "Beleza! Em alguns instantes você deve receber um e-mail com um link para redefinir sua senha.\nPode verificar na sua caixa de entrada (ou na pasta de spam, caso não apareça logo)?",
                        'expected_responses': {
                            'sim': 'click_link',
                            'yes': 'click_link',
                            'recebi': 'click_link',
                            'chegou': 'click_link',
                            'nao': 'email_troubleshoot',
                            'não': 'email_troubleshoot',
                            'no': 'email_troubleshoot',
                            'nao chegou': 'email_troubleshoot'
                        },
                        'fallback_message': "Você recebeu o e-mail de recuperação? Responda 'sim' se recebeu ou 'não' se ainda não chegou."
                    },
                    'email_troubleshoot': {
                        'message': "Sem problemas! Vamos verificar:\n1. Confira a pasta de spam/lixo eletrônico\n2. Aguarde mais alguns minutos (pode demorar até 10 min)\n3. Verifique se digitou o e-mail correto\n\nO e-mail chegou agora?",
                        'expected_responses': {
                            'sim': 'click_link',
                            'yes': 'click_link',
                            'recebi': 'click_link',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "O e-mail de recuperação chegou? Responda 'sim' ou 'não'."
                    },
                    'click_link': {
                        'message': "Maravilha 🎉\nClique no link e escolha uma nova senha.\nDica: use uma senha com pelo menos 8 caracteres, incluindo números e letras para ficar mais segura.",
                        'expected_responses': {
                            'sim': 'test_login',
                            'yes': 'test_login',
                            'pronto': 'test_login',
                            'feito': 'test_login',
                            'redefinida': 'test_login',
                            'alterada': 'test_login'
                        },
                        'fallback_message': "Você já redefiniu sua senha? Responda quando tiver criado a nova senha."
                    },
                    'test_login': {
                        'message': "Perfeito! 🚀\nAgora tente fazer login novamente com a nova senha. Conseguiu acessar sua conta?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'consegui': 'success',
                            'funcionou': 'success',
                            'entrei': 'success',
                            'nao': 'login_troubleshoot',
                            'não': 'login_troubleshoot',
                            'no': 'login_troubleshoot'
                        },
                        'fallback_message': "Conseguiu fazer login com a nova senha? Responda 'sim' ou 'não'."
                    },
                    'login_troubleshoot': {
                        'message': "Vamos verificar:\n1. Certifique-se de que está digitando a senha correta\n2. Verifique se o Caps Lock não está ativado\n3. Tente copiar e colar a senha\n\nFuncionou agora?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu fazer login agora? Responda 'sim' ou 'não'."
                    },
                    'different_problem': {
                        'message': "Entendi. Qual é o problema específico com sua senha?\n• Esqueci a senha\n• A senha não funciona\n• Conta bloqueada\n• Outro problema",
                        'expected_responses': {
                            'esqueci': 'access_login',
                            'nao funciona': 'login_troubleshoot',
                            'bloqueada': 'escalate_support',
                            'outro': 'escalate_support'
                        },
                        'fallback_message': "Pode me dizer qual é o problema específico com sua senha?"
                    },
                    'escalate_support': {
                        'message': "Entendo que precisa de uma ajuda mais específica. Vou te conectar com um atendente humano que poderá ajudar melhor com seu caso.\n\nEnquanto isso, você pode tentar:\n• Entrar em contato com o suporte técnico\n• Verificar se há atualizações do sistema\n• Tentar em outro navegador",
                        'expected_responses': {},
                        'solution': "Caso escalado para suporte humano. Orientações básicas fornecidas."
                    },
                    'success': {
                        'message': "Excelente! 🎊 Sua senha foi redefinida com sucesso e você conseguiu acessar sua conta.\n\nPosso ajudar com mais alguma coisa?",
                        'expected_responses': {},
                        'solution': "Senha redefinida com sucesso. Usuário conseguiu acessar a conta."
                    }
                }
            },
            
            'wifi_troubleshooting': {
                'steps': {
                    'start': {
                        'message': "Olá! Vejo que você está com problemas de Wi-Fi. Vamos resolver isso juntos! 📶\n\nPrimeiro, me diga: você consegue ver sua rede Wi-Fi na lista de redes disponíveis?",
                        'expected_responses': {
                            'sim': 'check_password',
                            'yes': 'check_password',
                            'vejo': 'check_password',
                            'aparece': 'check_password',
                            'nao': 'check_router',
                            'não': 'check_router',
                            'no': 'check_router',
                            'nao aparece': 'check_router'
                        },
                        'fallback_message': "Você consegue ver sua rede Wi-Fi na lista de redes disponíveis do seu dispositivo? Responda 'sim' ou 'não'."
                    },
                    'check_password': {
                        'message': "Ótimo! A rede aparece na lista. Quando você tenta conectar, o que acontece?\n\n• Pede senha e não conecta\n• Conecta mas não navega\n• Fica tentando conectar\n• Outro problema",
                        'expected_responses': {
                            'senha': 'wrong_password',
                            'pede senha': 'wrong_password',
                            'nao conecta': 'wrong_password',
                            'conecta': 'connected_no_internet',
                            'navega': 'connected_no_internet',
                            'internet': 'connected_no_internet',
                            'tentando': 'connection_timeout',
                            'outro': 'other_wifi_problem'
                        },
                        'fallback_message': "O que acontece quando você tenta conectar na rede Wi-Fi? Pode descrever o problema?"
                    },
                    'wrong_password': {
                        'message': "Parece ser um problema de senha. Vamos verificar:\n\n1. A senha do Wi-Fi geralmente está na etiqueta do roteador\n2. Pode estar escrita como 'Password', 'Key', 'WPA' ou 'Senha'\n3. Cuidado com letras maiúsculas e minúsculas\n\nVocê tem acesso ao roteador para verificar a senha?",
                        'expected_responses': {
                            'sim': 'check_router_label',
                            'yes': 'check_router_label',
                            'tenho': 'check_router_label',
                            'nao': 'ask_admin',
                            'não': 'ask_admin',
                            'no': 'ask_admin'
                        },
                        'fallback_message': "Você tem acesso físico ao roteador para verificar a senha na etiqueta? Responda 'sim' ou 'não'."
                    },
                    'check_router_label': {
                        'message': "Perfeito! Procure na parte de trás ou embaixo do roteador por uma etiqueta com:\n• Password\n• WPA Key\n• Senha Wi-Fi\n• Network Key\n\nEncontrou a senha na etiqueta?",
                        'expected_responses': {
                            'sim': 'try_new_password',
                            'yes': 'try_new_password',
                            'encontrei': 'try_new_password',
                            'achei': 'try_new_password',
                            'nao': 'reset_router_option',
                            'não': 'reset_router_option',
                            'no': 'reset_router_option'
                        },
                        'fallback_message': "Conseguiu encontrar a senha na etiqueta do roteador? Responda 'sim' ou 'não'."
                    },
                    'try_new_password': {
                        'message': "Ótimo! Agora:\n1. Vá nas configurações de Wi-Fi do seu dispositivo\n2. Clique na sua rede\n3. Digite a senha exatamente como está na etiqueta\n4. Tente conectar\n\nConseguiu conectar?",
                        'expected_responses': {
                            'sim': 'test_internet',
                            'yes': 'test_internet',
                            'conectou': 'test_internet',
                            'funcionou': 'test_internet',
                            'nao': 'password_troubleshoot',
                            'não': 'password_troubleshoot',
                            'no': 'password_troubleshoot'
                        },
                        'fallback_message': "Conseguiu conectar na rede Wi-Fi com a senha da etiqueta? Responda 'sim' ou 'não'."
                    },
                    'password_troubleshoot': {
                        'message': "Vamos tentar algumas coisas:\n1. Verifique se não há espaços antes ou depois da senha\n2. Confirme maiúsculas e minúsculas\n3. Alguns caracteres podem ser confusos (0 vs O, 1 vs l)\n\nTente novamente. Funcionou?",
                        'expected_responses': {
                            'sim': 'test_internet',
                            'yes': 'test_internet',
                            'funcionou': 'test_internet',
                            'nao': 'reset_router_option',
                            'não': 'reset_router_option',
                            'no': 'reset_router_option'
                        },
                        'fallback_message': "Conseguiu conectar agora? Responda 'sim' ou 'não'."
                    },
                    'ask_admin': {
                        'message': "Sem problemas! Você precisa perguntar a senha para:\n• Quem configurou o Wi-Fi\n• Administrador da rede\n• Responsável pela internet\n\nOu posso te ajudar a resetar o roteador (isso vai criar uma nova senha). O que prefere?",
                        'expected_responses': {
                            'perguntar': 'wait_password',
                            'admin': 'wait_password',
                            'resetar': 'reset_router_option',
                            'reset': 'reset_router_option',
                            'nova senha': 'reset_router_option'
                        },
                        'fallback_message': "Você vai perguntar a senha para alguém ou prefere resetar o roteador?"
                    },
                    'wait_password': {
                        'message': "Perfeito! Quando conseguir a senha, volte aqui que eu te ajudo a conectar.\n\nEnquanto isso, você pode:\n• Anotar a senha corretamente\n• Verificar se o dispositivo está próximo do roteador\n• Reiniciar o Wi-Fi do seu dispositivo\n\nConseguiu a senha?",
                        'expected_responses': {
                            'sim': 'try_new_password',
                            'yes': 'try_new_password',
                            'consegui': 'try_new_password',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu a senha do Wi-Fi? Responda 'sim' quando tiver ou 'não' se precisar de outra solução."
                    },
                    'check_router': {
                        'message': "A rede não aparece na lista. Vamos verificar o roteador:\n\n1. O roteador está ligado? (luzes acesas)\n2. Os cabos estão bem conectados?\n3. Há energia elétrica?\n\nO roteador está ligado e com luzes acesas?",
                        'expected_responses': {
                            'sim': 'restart_router',
                            'yes': 'restart_router',
                            'ligado': 'restart_router',
                            'luzes': 'restart_router',
                            'nao': 'power_check',
                            'não': 'power_check',
                            'no': 'power_check',
                            'desligado': 'power_check'
                        },
                        'fallback_message': "O roteador está ligado com luzes acesas? Responda 'sim' ou 'não'."
                    },
                    'power_check': {
                        'message': "Vamos verificar a alimentação:\n1. O cabo de energia está conectado?\n2. A tomada está funcionando?\n3. O botão liga/desliga está acionado?\n\nTente ligar o roteador. Acendeu alguma luz?",
                        'expected_responses': {
                            'sim': 'restart_router',
                            'yes': 'restart_router',
                            'acendeu': 'restart_router',
                            'ligou': 'restart_router',
                            'nao': 'power_troubleshoot',
                            'não': 'power_troubleshoot',
                            'no': 'power_troubleshoot'
                        },
                        'fallback_message': "O roteador ligou e acendeu alguma luz? Responda 'sim' ou 'não'."
                    },
                    'power_troubleshoot': {
                        'message': "Problema de energia. Vamos resolver:\n1. Teste a tomada com outro aparelho\n2. Verifique se o cabo não está danificado\n3. Procure o botão liga/desliga no roteador\n\nSe nada funcionar, pode ser problema no roteador. Conseguiu ligar?",
                        'expected_responses': {
                            'sim': 'restart_router',
                            'yes': 'restart_router',
                            'funcionou': 'restart_router',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu ligar o roteador? Responda 'sim' ou 'não'."
                    },
                    'restart_router': {
                        'message': "Ótimo! Agora vamos reiniciar o roteador:\n1. Desligue o roteador da tomada\n2. Aguarde 30 segundos\n3. Ligue novamente\n4. Aguarde 2-3 minutos para estabilizar\n\nFez o reinício? As luzes estabilizaram?",
                        'expected_responses': {
                            'sim': 'check_network_again',
                            'yes': 'check_network_again',
                            'reiniciei': 'check_network_again',
                            'estabilizou': 'check_network_again',
                            'nao': 'wait_longer',
                            'não': 'wait_longer',
                            'no': 'wait_longer'
                        },
                        'fallback_message': "Você reiniciou o roteador e as luzes estabilizaram? Responda 'sim' ou 'não'."
                    },
                    'wait_longer': {
                        'message': "Sem pressa! O roteador pode demorar até 5 minutos para estabilizar completamente.\n\nEnquanto aguarda, verifique se:\n• A luz de energia está fixa (não piscando)\n• A luz de internet está acesa\n• A luz de Wi-Fi está ativa\n\nAgora as luzes estão estáveis?",
                        'expected_responses': {
                            'sim': 'check_network_again',
                            'yes': 'check_network_again',
                            'estaveis': 'check_network_again',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "As luzes do roteador estão estáveis agora? Responda 'sim' ou 'não'."
                    },
                    'check_network_again': {
                        'message': "Perfeito! Agora vamos verificar se a rede apareceu:\n1. Vá nas configurações de Wi-Fi do seu dispositivo\n2. Atualize a lista de redes\n3. Procure pelo nome da sua rede\n\nSua rede Wi-Fi aparece na lista agora?",
                        'expected_responses': {
                            'sim': 'check_password',
                            'yes': 'check_password',
                            'aparece': 'check_password',
                            'vejo': 'check_password',
                            'nao': 'network_name_help',
                            'não': 'network_name_help',
                            'no': 'network_name_help'
                        },
                        'fallback_message': "Sua rede Wi-Fi aparece na lista agora? Responda 'sim' ou 'não'."
                    },
                    'network_name_help': {
                        'message': "Vamos encontrar sua rede:\n• O nome pode estar na etiqueta do roteador\n• Procure por 'SSID', 'Network Name' ou 'Nome da Rede'\n• Pode ser algo como 'NET_2G', 'Vivo-Fibra', etc.\n\nQual nome você vê na etiqueta do roteador?",
                        'expected_responses': {
                            'encontrei': 'try_connect_network',
                            'achei': 'try_connect_network',
                            'vejo': 'try_connect_network',
                            'nao tem': 'escalate_support',
                            'não tem': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu encontrar o nome da rede na etiqueta? Responda com o nome ou 'não tem' se não encontrar."
                    },
                    'try_connect_network': {
                        'message': "Ótimo! Agora:\n1. Procure esse nome na lista de redes Wi-Fi\n2. Clique nele para conectar\n3. Digite a senha (também na etiqueta)\n\nConseguiu conectar?",
                        'expected_responses': {
                            'sim': 'test_internet',
                            'yes': 'test_internet',
                            'conectou': 'test_internet',
                            'nao': 'final_troubleshoot',
                            'não': 'final_troubleshoot',
                            'no': 'final_troubleshoot'
                        },
                        'fallback_message': "Conseguiu conectar na rede Wi-Fi? Responda 'sim' ou 'não'."
                    },
                    'connected_no_internet': {
                        'message': "Você está conectado ao Wi-Fi mas sem internet. Vamos resolver:\n\n1. Teste abrir um site (google.com)\n2. Reinicie o navegador\n3. Verifique se outros dispositivos têm internet\n\nOutros dispositivos (celular, TV) têm internet na mesma rede?",
                        'expected_responses': {
                            'sim': 'device_problem',
                            'yes': 'device_problem',
                            'tem': 'device_problem',
                            'funcionam': 'device_problem',
                            'nao': 'internet_provider_issue',
                            'não': 'internet_provider_issue',
                            'no': 'internet_provider_issue'
                        },
                        'fallback_message': "Outros dispositivos têm internet na mesma rede Wi-Fi? Responda 'sim' ou 'não'."
                    },
                    'device_problem': {
                        'message': "O problema é específico do seu dispositivo. Vamos resolver:\n1. Esqueça a rede Wi-Fi e conecte novamente\n2. Reinicie seu dispositivo\n3. Verifique se há atualizações pendentes\n\nTente essas soluções. Funcionou?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "As soluções funcionaram? Responda 'sim' ou 'não'."
                    },
                    'internet_provider_issue': {
                        'message': "Parece ser um problema com seu provedor de internet. Vamos verificar:\n1. Reinicie o modem (se for separado do roteador)\n2. Verifique se há manutenção na região\n3. Entre em contato com seu provedor\n\nO reinício do modem resolveu?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'nao': 'contact_provider',
                            'não': 'contact_provider',
                            'no': 'contact_provider'
                        },
                        'fallback_message': "O reinício do modem resolveu o problema? Responda 'sim' ou 'não'."
                    },
                    'contact_provider': {
                        'message': "Você precisa entrar em contato com seu provedor de internet:\n• Informe que está sem internet\n• Mencione que já reiniciou os equipamentos\n• Pergunte sobre manutenções na região\n\nO número geralmente está na conta ou no roteador.",
                        'expected_responses': {},
                        'solution': "Problema de provedor de internet. Usuário orientado a entrar em contato."
                    },
                    'connection_timeout': {
                        'message': "O dispositivo fica tentando conectar. Isso pode ser:\n• Senha incorreta\n• Sinal fraco\n• Problema no roteador\n\nVocê está próximo do roteador (mesma sala)?",
                        'expected_responses': {
                            'sim': 'wrong_password',
                            'yes': 'wrong_password',
                            'proximo': 'wrong_password',
                            'nao': 'signal_strength',
                            'não': 'signal_strength',
                            'no': 'signal_strength',
                            'longe': 'signal_strength'
                        },
                        'fallback_message': "Você está próximo do roteador? Responda 'sim' ou 'não'."
                    },
                    'signal_strength': {
                        'message': "O sinal pode estar fraco. Vamos melhorar:\n1. Aproxime-se do roteador\n2. Remova obstáculos (paredes, móveis)\n3. Verifique se há interferências (micro-ondas, outros roteadores)\n\nTente conectar mais próximo do roteador. Funcionou?",
                        'expected_responses': {
                            'sim': 'test_internet',
                            'yes': 'test_internet',
                            'funcionou': 'test_internet',
                            'nao': 'wrong_password',
                            'não': 'wrong_password',
                            'no': 'wrong_password'
                        },
                        'fallback_message': "Conseguiu conectar mais próximo do roteador? Responda 'sim' ou 'não'."
                    },
                    'test_internet': {
                        'message': "Excelente! Você está conectado. Vamos testar a internet:\n1. Abra um navegador\n2. Acesse google.com ou youtube.com\n3. Teste a velocidade\n\nA internet está funcionando normalmente?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionando': 'success',
                            'normal': 'success',
                            'lenta': 'speed_troubleshoot',
                            'devagar': 'speed_troubleshoot',
                            'nao': 'connected_no_internet',
                            'não': 'connected_no_internet',
                            'no': 'connected_no_internet'
                        },
                        'fallback_message': "A internet está funcionando? Responda 'sim', 'lenta' ou 'não'."
                    },
                    'speed_troubleshoot': {
                        'message': "Internet lenta pode ter várias causas:\n1. Muitos dispositivos conectados\n2. Downloads em andamento\n3. Problema no provedor\n4. Roteador sobrecarregado\n\nTente reiniciar o roteador novamente. Melhorou a velocidade?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'melhorou': 'success',
                            'nao': 'contact_provider',
                            'não': 'contact_provider',
                            'no': 'contact_provider'
                        },
                        'fallback_message': "A velocidade melhorou após reiniciar? Responda 'sim' ou 'não'."
                    },
                    'reset_router_option': {
                        'message': "Podemos resetar o roteador para configurar uma nova senha:\n⚠️ ATENÇÃO: Isso vai apagar todas as configurações!\n\n1. Procure o botão 'Reset' no roteador\n2. Mantenha pressionado por 10 segundos\n3. Aguarde reiniciar\n\nQuer fazer o reset?",
                        'expected_responses': {
                            'sim': 'reset_instructions',
                            'yes': 'reset_instructions',
                            'quero': 'reset_instructions',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Quer fazer o reset do roteador? Responda 'sim' ou 'não'."
                    },
                    'reset_instructions': {
                        'message': "Instruções para reset:\n1. Com o roteador ligado, encontre o botão 'Reset'\n2. Use um clipe ou palito para pressionar\n3. Mantenha pressionado por 10-15 segundos\n4. Solte e aguarde 2-3 minutos\n\nFez o reset? As luzes voltaram ao normal?",
                        'expected_responses': {
                            'sim': 'post_reset_config',
                            'yes': 'post_reset_config',
                            'fiz': 'post_reset_config',
                            'nao': 'reset_help',
                            'não': 'reset_help',
                            'no': 'reset_help'
                        },
                        'fallback_message': "Conseguiu fazer o reset? Responda 'sim' ou 'não'."
                    },
                    'post_reset_config': {
                        'message': "Perfeito! Após o reset, o roteador volta às configurações de fábrica.\n\nProcure na etiqueta por:\n• Nome padrão da rede (SSID)\n• Senha padrão\n\nGeralmente algo como 'admin', '12345678' ou está na etiqueta.\n\nConseguiu conectar com as configurações padrão?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'conectei': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu conectar com as configurações padrão? Responda 'sim' ou 'não'."
                    },
                    'other_wifi_problem': {
                        'message': "Pode me descrever melhor o problema? Por exemplo:\n• Mensagem de erro específica\n• O que acontece quando tenta conectar\n• Há quanto tempo não funciona\n\nIsso me ajudará a encontrar a melhor solução.",
                        'expected_responses': {
                            'erro': 'error_analysis',
                            'mensagem': 'error_analysis',
                            'nao conecta': 'connection_timeout',
                            'lento': 'speed_troubleshoot',
                            'cai': 'connection_drops'
                        },
                        'fallback_message': "Pode descrever melhor o problema com o Wi-Fi?"
                    },
                    'final_troubleshoot': {
                        'message': "Vamos tentar as últimas soluções:\n1. Esqueça a rede e reconecte\n2. Reinicie seu dispositivo\n3. Verifique se há atualizações\n4. Teste com outro dispositivo\n\nAlguma dessas soluções funcionou?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Alguma das soluções funcionou? Responda 'sim' ou 'não'."
                    },
                    'escalate_support': {
                        'message': "Entendo que o problema é mais complexo. Vou te conectar com um técnico especializado.\n\nEnquanto aguarda:\n• Anote o modelo do seu roteador\n• Verifique se há luzes piscando\n• Teste com outros dispositivos\n\nUm técnico entrará em contato em breve.",
                        'expected_responses': {},
                        'solution': "Caso escalado para suporte técnico especializado."
                    },
                    'success': {
                        'message': "Fantástico! 🎉 Seu Wi-Fi está funcionando perfeitamente!\n\nDicas para manter a conexão estável:\n• Mantenha o roteador em local ventilado\n• Reinicie mensalmente\n• Mantenha firmware atualizado\n\nPosso ajudar com mais alguma coisa?",
                        'expected_responses': {},
                        'solution': "Wi-Fi configurado e funcionando com sucesso."
                    }
                }
            },
            
            'printer_troubleshooting': {
                'steps': {
                    'start': {
                        'message': "Olá! Vejo que você está com problemas na impressora. Vamos resolver isso! 🖨️\n\nPrimeiro, me diga: qual é o problema específico?\n• Não imprime nada\n• Imprime com qualidade ruim\n• Papel atolado\n• Não reconhece a impressora\n• Outro problema",
                        'expected_responses': {
                            'nao imprime': 'check_power',
                            'não imprime': 'check_power',
                            'nada': 'check_power',
                            'qualidade': 'print_quality',
                            'ruim': 'print_quality',
                            'papel': 'paper_jam',
                            'atolado': 'paper_jam',
                            'nao reconhece': 'connection_issue',
                            'não reconhece': 'connection_issue',
                            'reconhece': 'connection_issue',
                            'outro': 'other_printer_problem'
                        },
                        'fallback_message': "Qual é o problema específico com sua impressora? Pode escolher uma das opções ou descrever o problema."
                    },
                    'check_power': {
                        'message': "Vamos verificar o básico primeiro:\n\n1. A impressora está ligada? (luzes acesas)\n2. O cabo de energia está conectado?\n3. Há papel na bandeja?\n4. Há tinta/toner suficiente?\n\nA impressora está ligada com luzes acesas?",
                        'expected_responses': {
                            'sim': 'check_connection',
                            'yes': 'check_connection',
                            'ligada': 'check_connection',
                            'luzes': 'check_connection',
                            'nao': 'power_troubleshoot',
                            'não': 'power_troubleshoot',
                            'no': 'power_troubleshoot',
                            'desligada': 'power_troubleshoot'
                        },
                        'fallback_message': "A impressora está ligada com luzes acesas? Responda 'sim' ou 'não'."
                    },
                    'power_troubleshoot': {
                        'message': "Vamos ligar a impressora:\n1. Verifique se o cabo está bem conectado\n2. Teste a tomada com outro aparelho\n3. Procure o botão liga/desliga\n4. Pressione firmemente o botão\n\nConseguiu ligar a impressora?",
                        'expected_responses': {
                            'sim': 'check_connection',
                            'yes': 'check_connection',
                            'ligou': 'check_connection',
                            'funcionou': 'check_connection',
                            'nao': 'power_issue',
                            'não': 'power_issue',
                            'no': 'power_issue'
                        },
                        'fallback_message': "Conseguiu ligar a impressora? Responda 'sim' ou 'não'."
                    },
                    'power_issue': {
                        'message': "Problema de energia na impressora:\n• Cabo de energia danificado\n• Problema na tomada\n• Defeito interno\n\nTeste com outro cabo de energia se tiver. Se não ligar, pode precisar de assistência técnica.\n\nVai tentar assistência técnica ou tem outro cabo para testar?",
                        'expected_responses': {
                            'assistencia': 'escalate_support',
                            'tecnica': 'escalate_support',
                            'cabo': 'test_cable',
                            'outro': 'test_cable'
                        },
                        'fallback_message': "Vai procurar assistência técnica ou tem outro cabo para testar?"
                    },
                    'test_cable': {
                        'message': "Ótimo! Teste com outro cabo de energia:\n1. Desligue a impressora\n2. Troque o cabo\n3. Conecte novamente\n4. Tente ligar\n\nFuncionou com o outro cabo?",
                        'expected_responses': {
                            'sim': 'check_connection',
                            'yes': 'check_connection',
                            'funcionou': 'check_connection',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "A impressora ligou com o outro cabo? Responda 'sim' ou 'não'."
                    },
                    'check_connection': {
                        'message': "Ótimo! A impressora está ligada. Agora vamos verificar a conexão:\n\nComo sua impressora está conectada?\n• Cabo USB\n• Wi-Fi\n• Cabo de rede (Ethernet)\n• Bluetooth",
                        'expected_responses': {
                            'usb': 'check_usb',
                            'cabo': 'check_usb',
                            'wifi': 'check_wifi_printer',
                            'wi-fi': 'check_wifi_printer',
                            'rede': 'check_ethernet',
                            'ethernet': 'check_ethernet',
                            'bluetooth': 'check_bluetooth'
                        },
                        'fallback_message': "Como sua impressora está conectada ao computador? USB, Wi-Fi, cabo de rede ou Bluetooth?"
                    },
                    'check_usb': {
                        'message': "Conexão USB. Vamos verificar:\n1. O cabo USB está bem conectado nos dois lados?\n2. Teste em outra porta USB do computador\n3. O computador reconhece a impressora?\n\nO computador mostra que a impressora está conectada?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'reconhece': 'test_print',
                            'mostra': 'test_print',
                            'nao': 'usb_troubleshoot',
                            'não': 'usb_troubleshoot',
                            'no': 'usb_troubleshoot'
                        },
                        'fallback_message': "O computador reconhece a impressora conectada por USB? Responda 'sim' ou 'não'."
                    },
                    'usb_troubleshoot': {
                        "message": "Vamos resolver a conexão USB:\n1. Troque de porta USB\n2. Teste outro cabo USB se tiver\n3. Reinicie o computador com a impressora conectada\n4. Verifique se precisa instalar drivers\n\nTentou essas soluções? Funcionou alguma?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'funcionou': 'test_print',
                            'nao': 'driver_install',
                            'não': 'driver_install',
                            'no': 'driver_install'
                        },
                        'fallback_message': "Alguma das soluções USB funcionou? Responda 'sim' ou 'não'."
                    },
                    'check_wifi_printer': {
                        'message': "Impressora Wi-Fi. Vamos verificar:\n1. A impressora está conectada na mesma rede Wi-Fi?\n2. O computador está na mesma rede?\n3. A impressora aparece na lista de dispositivos?\n\nA impressora está na mesma rede Wi-Fi que o computador?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'mesma': 'test_print',
                            'rede': 'test_print',
                            'nao': 'wifi_printer_setup',
                            'não': 'wifi_printer_setup',
                            'no': 'wifi_printer_setup',
                            'nao sei': 'wifi_printer_setup'
                        },
                        'fallback_message': "A impressora está conectada na mesma rede Wi-Fi que o computador? Responda 'sim' ou 'não'."
                    },
                    'wifi_printer_setup': {
                        'message': "Vamos conectar a impressora no Wi-Fi:\n1. No painel da impressora, procure 'Configurações' ou 'Setup'\n2. Encontre 'Wi-Fi' ou 'Wireless'\n3. Selecione sua rede\n4. Digite a senha do Wi-Fi\n\nConseguiu encontrar as configurações de Wi-Fi na impressora?",
                        'expected_responses': {
                            'sim': 'wifi_connect_printer',
                            'yes': 'wifi_connect_printer',
                            'encontrei': 'wifi_connect_printer',
                            'achei': 'wifi_connect_printer',
                            'nao': 'wifi_printer_help',
                            'não': 'wifi_printer_help',
                            'no': 'wifi_printer_help'
                        },
                        'fallback_message': "Conseguiu encontrar as configurações de Wi-Fi na impressora? Responda 'sim' ou 'não'."
                    },
                    'wifi_connect_printer': {
                        'message': "Perfeito! Agora:\n1. Selecione sua rede Wi-Fi na lista\n2. Digite a senha (mesma do seu computador/celular)\n3. Confirme a conexão\n4. Aguarde a confirmação\n\nA impressora conectou no Wi-Fi? (geralmente mostra um ícone ou mensagem)",
                        'expected_responses': {
                            'sim': 'add_printer_computer',
                            'yes': 'add_printer_computer',
                            'conectou': 'add_printer_computer',
                            'funcionou': 'add_printer_computer',
                            'nao': 'wifi_password_help',
                            'não': 'wifi_password_help',
                            'no': 'wifi_password_help'
                        },
                        'fallback_message': "A impressora conectou no Wi-Fi? Responda 'sim' ou 'não'."
                    },
                    'wifi_password_help': {
                        'message': "Problema na conexão Wi-Fi da impressora:\n1. Verifique se a senha está correta\n2. Certifique-se de que está na rede 2.4GHz (não 5GHz)\n3. Aproxime a impressora do roteador\n4. Reinicie a impressora e tente novamente\n\nTentou novamente? Funcionou?",
                        'expected_responses': {
                            'sim': 'add_printer_computer',
                            'yes': 'add_printer_computer',
                            'funcionou': 'add_printer_computer',
                            'nao': 'wifi_printer_help',
                            'não': 'wifi_printer_help',
                            'no': 'wifi_printer_help'
                        },
                        'fallback_message': "Conseguiu conectar a impressora no Wi-Fi agora? Responda 'sim' ou 'não'."
                    },
                    'wifi_printer_help': {
                        'message': "Algumas impressoras têm métodos alternativos:\n• Botão WPS (pressione no roteador e na impressora)\n• Aplicativo do fabricante (HP Smart, Canon PRINT, etc.)\n• Configuração via cabo USB temporário\n\nQual método quer tentar?",
                        'expected_responses': {
                            'wps': 'wps_setup',
                            'aplicativo': 'app_setup',
                            'app': 'app_setup',
                            'cabo': 'usb_temp_setup',
                            'usb': 'usb_temp_setup'
                        },
                        'fallback_message': "Qual método quer tentar: WPS, aplicativo do fabricante ou cabo USB temporário?"
                    },
                    'add_printer_computer': {
                        'message': "Ótimo! A impressora está conectada no Wi-Fi. Agora vamos adicioná-la ao computador:\n\nWindows:\n1. Configurações > Impressoras e scanners\n2. Adicionar impressora\n3. Selecione sua impressora\n\nMac:\n1. Preferências > Impressoras\n2. Clique no +\n3. Selecione sua impressora\n\nConseguiu adicionar?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'adicionei': 'test_print',
                            'funcionou': 'test_print',
                            'nao': 'add_printer_help',
                            'não': 'add_printer_help',
                            'no': 'add_printer_help'
                        },
                        'fallback_message': "Conseguiu adicionar a impressora no computador? Responda 'sim' ou 'não'."
                    },
                    'driver_install': {
                        'message': "Vamos instalar os drivers da impressora:\n1. Acesse o site do fabricante (HP, Canon, Epson, etc.)\n2. Procure por 'Suporte' ou 'Downloads'\n3. Digite o modelo da sua impressora\n4. Baixe e instale o driver\n\nQual é a marca da sua impressora?",
                        'expected_responses': {
                            'hp': 'hp_driver',
                            'canon': 'canon_driver',
                            'epson': 'epson_driver',
                            'brother': 'brother_driver',
                            'samsung': 'samsung_driver'
                        },
                        'fallback_message': "Qual é a marca da sua impressora? HP, Canon, Epson, Brother, Samsung ou outra?"
                    },
                    'test_print': {
                        'message': "Excelente! Agora vamos testar a impressão:\n1. Abra um documento simples (Bloco de Notas)\n2. Digite algumas palavras\n3. Clique em 'Imprimir' ou Ctrl+P\n4. Selecione sua impressora\n5. Clique em 'Imprimir'\n\nA impressora imprimiu o teste?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'imprimiu': 'success',
                            'funcionou': 'success',
                            'nao': 'print_troubleshoot',
                            'não': 'print_troubleshoot',
                            'no': 'print_troubleshoot'
                        },
                        'fallback_message': "A impressora imprimiu o teste? Responda 'sim' ou 'não'."
                    },
                    'print_troubleshoot': {
                        'message': "Vamos resolver o problema de impressão:\n1. Verifique se há papel na bandeja\n2. Confirme se há tinta/toner\n3. Verifique se não há papel atolado\n4. Reinicie a impressora\n\nO que você observa? Há alguma luz piscando ou mensagem de erro?",
                        'expected_responses': {
                            'papel': 'paper_issue',
                            'tinta': 'ink_issue',
                            'atolado': 'paper_jam',
                            'erro': 'error_message',
                            'luz': 'error_lights',
                            'nada': 'general_troubleshoot'
                        },
                        'fallback_message': "O que você observa na impressora? Papel, tinta, atolamento, erro ou nada específico?"
                    },
                    'paper_issue': {
                        'message': "Problema com papel:\n1. Verifique se há papel suficiente na bandeja\n2. Ajuste as guias laterais do papel\n3. Use papel do tamanho correto (A4, Carta)\n4. Não sobrecarregue a bandeja\n\nColocou papel corretamente? Tente imprimir novamente.",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'funcionou': 'test_print',
                            'nao': 'paper_jam',
                            'não': 'paper_jam',
                            'no': 'paper_jam'
                        },
                        'fallback_message': "Colocou o papel corretamente? Responda 'sim' ou 'não'."
                    },
                    'ink_issue': {
                        'message': "Problema com tinta/toner:\n1. Verifique o nível de tinta no painel ou computador\n2. Remova e recoloque os cartuchos\n3. Limpe os contatos dos cartuchos\n4. Se necessário, substitua cartuchos vazios\n\nOs cartuchos têm tinta suficiente?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'tem': 'test_print',
                            'suficiente': 'test_print',
                            'nao': 'replace_cartridge',
                            'não': 'replace_cartridge',
                            'no': 'replace_cartridge',
                            'vazio': 'replace_cartridge'
                        },
                        'fallback_message': "Os cartuchos têm tinta suficiente? Responda 'sim' ou 'não'."
                    },
                    'replace_cartridge': {
                        'message': "Hora de trocar os cartuchos:\n1. Abra a tampa da impressora\n2. Remova o cartucho vazio\n3. Desembale o novo cartucho\n4. Remova todas as fitas protetoras\n5. Instale o novo cartucho\n\nTrocou o cartucho? Tente imprimir novamente.",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'troquei': 'test_print',
                            'instalei': 'test_print',
                            'nao': 'cartridge_help',
                            'não': 'cartridge_help',
                            'no': 'cartridge_help'
                        },
                        'fallback_message': "Conseguiu trocar o cartucho? Responda 'sim' ou 'não'."
                    },
                    'paper_jam': {
                        'message': "Vamos resolver o papel atolado:\n1. Desligue a impressora\n2. Abra todas as tampas\n3. Remova cuidadosamente o papel atolado\n4. Verifique se não sobrou pedaços\n5. Feche as tampas e ligue novamente\n\nConseguiu remover todo o papel atolado?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'removi': 'test_print',
                            'limpei': 'test_print',
                            'nao': 'paper_jam_help',
                            'não': 'paper_jam_help',
                            'no': 'paper_jam_help'
                        },
                        'fallback_message': "Conseguiu remover todo o papel atolado? Responda 'sim' ou 'não'."
                    },
                    'paper_jam_help': {
                        'message': "Para papel atolado difícil:\n1. Use uma lanterna para ver melhor\n2. Puxe o papel na direção do movimento\n3. Não force, pode danificar a impressora\n4. Se necessário, consulte o manual\n\nSe não conseguir, pode precisar de assistência técnica. Conseguiu agora?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'consegui': 'test_print',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        },
                        'fallback_message': "Conseguiu remover o papel atolado? Responda 'sim' ou 'não'."
                    },
                    'print_quality': {
                        'message': "Problema de qualidade de impressão. Qual é o problema específico?\n• Texto borrado ou manchado\n• Cores desbotadas\n• Linhas ou riscos\n• Impressão muito clara\n• Impressão cortada",
                        'expected_responses': {
                            'borrado': 'clean_heads',
                            'manchado': 'clean_heads',
                            'cores': 'color_issue',
                            'desbotadas': 'color_issue',
                            'linhas': 'alignment_issue',
                            'riscos': 'alignment_issue',
                            'clara': 'ink_issue',
                            'cortada': 'paper_size_issue'
                        },
                        'fallback_message': "Qual é o problema específico de qualidade? Borrado, cores ruins, linhas, muito claro ou cortado?"
                    },
                    'clean_heads': {
                        'message': "Vamos limpar os cabeçotes de impressão:\n1. Acesse as configurações da impressora\n2. Procure por 'Manutenção' ou 'Limpeza'\n3. Execute 'Limpeza dos cabeçotes'\n4. Aguarde o processo terminar\n5. Imprima uma página de teste\n\nA qualidade melhorou após a limpeza?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'melhorou': 'success',
                            'nao': 'deep_clean',
                            'não': 'deep_clean',
                            'no': 'deep_clean'
                        },
                        'fallback_message': "A qualidade de impressão melhorou após a limpeza? Responda 'sim' ou 'não'."
                    },
                    'deep_clean': {
                        'message': "Vamos fazer uma limpeza profunda:\n1. Execute 'Limpeza profunda' ou 'Deep cleaning'\n2. Aguarde (pode demorar alguns minutos)\n3. Imprima página de teste\n4. Se necessário, repita o processo\n\nA limpeza profunda resolveu?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'resolveu': 'success',
                            'nao': 'replace_cartridge',
                            'não': 'replace_cartridge',
                            'no': 'replace_cartridge'
                        },
                        'fallback_message': "A limpeza profunda resolveu o problema? Responda 'sim' ou 'não'."
                    },
                    'escalate_support': {
                        'message': "O problema parece mais complexo e pode precisar de assistência técnica especializada.\n\nAntes de procurar assistência:\n• Anote o modelo exato da impressora\n• Descreva o problema detalhadamente\n• Verifique se ainda está na garantia\n\nUm técnico poderá ajudar melhor com seu caso específico.",
                        'expected_responses': {},
                        'solution': "Caso escalado para assistência técnica especializada."
                    },
                    'success': {
                        'message': "Excelente! 🎉 Sua impressora está funcionando perfeitamente!\n\nDicas para manter a impressora em bom estado:\n• Imprima pelo menos uma página por semana\n• Mantenha cartuchos originais ou compatíveis de qualidade\n• Limpe regularmente\n• Use papel de boa qualidade\n\nPosso ajudar com mais alguma coisa?",
                        'expected_responses': {},
                        'solution': "Impressora configurada e funcionando com sucesso."
                    }
                }
            },
            
            'email_configuration': {
                'steps': {
                    'start': {
                        'message': "Olá! Vejo que você precisa de ajuda com configuração de email. 📧\n\nQual é sua situação?\n• Configurar email pela primeira vez\n• Email parou de funcionar\n• Não consigo enviar emails\n• Não recebo emails\n• Outro problema",
                        'expected_responses': {
                            'primeira': 'first_time_setup',
                            'primeira vez': 'first_time_setup',
                            'configurar': 'first_time_setup',
                            'parou': 'email_stopped',
                            'nao funciona': 'email_stopped',
                            'enviar': 'send_problem',
                            'nao consigo enviar': 'send_problem',
                            'receber': 'receive_problem',
                            'nao recebo': 'receive_problem',
                            'outro': 'other_email_problem'
                        },
                        'fallback_message': "Qual é o problema específico com seu email? Primeira configuração, parou de funcionar, problemas para enviar/receber ou outro?"
                    },
                    'first_time_setup': {
                        'message': "Vamos configurar seu email! Primeiro preciso saber:\n\nQual provedor de email você usa?\n• Gmail\n• Outlook/Hotmail\n• Yahoo\n• Email corporativo/trabalho\n• Outro provedor",
                        'expected_responses': {
                            'gmail': 'gmail_setup',
                            'google': 'gmail_setup',
                            'outlook': 'outlook_setup',
                            'hotmail': 'outlook_setup',
                            'yahoo': 'yahoo_setup',
                            'corporativo': 'corporate_setup',
                            'trabalho': 'corporate_setup',
                            'empresa': 'corporate_setup',
                            'outro': 'other_provider_setup'
                        },
                        'fallback_message': "Qual provedor de email você usa? Gmail, Outlook, Yahoo, email corporativo ou outro?"
                    },
                    'gmail_setup': {
                        'message': "Configuração do Gmail! 📬\n\nEm qual aplicativo você quer configurar?\n• Outlook (Windows/Mac)\n• Mail (iPhone/iPad)\n• Email (Android)\n• Thunderbird\n• Outro aplicativo",
                        'expected_responses': {
                            'outlook': 'gmail_outlook',
                            'mail': 'gmail_iphone',
                            'iphone': 'gmail_iphone',
                            'android': 'gmail_android',
                            'thunderbird': 'gmail_thunderbird',
                            'outro': 'gmail_generic'
                        },
                        'fallback_message': "Em qual aplicativo quer configurar o Gmail? Outlook, Mail (iPhone), Email (Android), Thunderbird ou outro?"
                    },
                    'gmail_outlook': {
                        'message': "Configurando Gmail no Outlook:\n\n1. Abra o Outlook\n2. Vá em Arquivo > Adicionar Conta\n3. Digite seu email do Gmail\n4. Clique em 'Conectar'\n5. Será redirecionado para login do Google\n\nConseguiu chegar na tela de login do Google?",
                        'expected_responses': {
                            'sim': 'gmail_outlook_login',
                            'yes': 'gmail_outlook_login',
                            'consegui': 'gmail_outlook_login',
                            'nao': 'gmail_outlook_help',
                            'não': 'gmail_outlook_help',
                            'no': 'gmail_outlook_help'
                        },
                        'fallback_message': "Conseguiu chegar na tela de login do Google? Responda 'sim' ou 'não'."
                    },
                    'gmail_outlook_login': {
                        'message': "Perfeito! Agora:\n1. Digite sua senha do Gmail\n2. Se tiver autenticação em 2 fatores, use o código\n3. Autorize o Outlook a acessar sua conta\n4. Aguarde a sincronização\n\nO Outlook conseguiu conectar e baixar seus emails?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'conectou': 'test_email_send',
                            'baixou': 'test_email_send',
                            'nao': 'gmail_auth_problem',
                            'não': 'gmail_auth_problem',
                            'no': 'gmail_auth_problem'
                        },
                        'fallback_message': "O Outlook conectou e baixou seus emails do Gmail? Responda 'sim' ou 'não'."
                    },
                    'gmail_auth_problem': {
                        'message': "Problema de autenticação. Vamos resolver:\n\n1. Verifique se a senha está correta\n2. Se usa autenticação em 2 fatores, pode precisar de senha de app\n3. Acesse myaccount.google.com\n4. Vá em Segurança > Senhas de app\n5. Gere uma senha específica para o Outlook\n\nQuer tentar gerar uma senha de app?",
                        'expected_responses': {
                            'sim': 'gmail_app_password',
                            'yes': 'gmail_app_password',
                            'quero': 'gmail_app_password',
                            'gerar': 'gmail_app_password',
                            'nao': 'gmail_basic_auth',
                            'não': 'gmail_basic_auth',
                            'no': 'gmail_basic_auth'
                        },
                        'fallback_message': "Quer tentar gerar uma senha de app para o Gmail? Responda 'sim' ou 'não'."
                    },
                    'gmail_app_password': {
                        'message': "Gerando senha de app:\n1. Acesse myaccount.google.com\n2. Segurança > Verificação em duas etapas\n3. Senhas de app\n4. Selecione 'Email' e 'Computador'\n5. Copie a senha gerada\n6. Use essa senha no Outlook (não sua senha normal)\n\nConseguiu gerar e usar a senha de app?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'funcionou': 'test_email_send',
                            'nao': 'gmail_basic_auth',
                            'não': 'gmail_basic_auth',
                            'no': 'gmail_basic_auth'
                        },
                        'fallback_message': "Conseguiu usar a senha de app? Responda 'sim' ou 'não'."
                    },
                    'outlook_setup': {
                        'message': "Configuração do Outlook/Hotmail! 📧\n\nEm qual aplicativo você quer configurar?\n• Outlook (Windows/Mac)\n• Mail (iPhone/iPad)\n• Email (Android)\n• Thunderbird\n• Outro aplicativo",
                        'expected_responses': {
                            'outlook': 'outlook_outlook',
                            'mail': 'outlook_iphone',
                            'iphone': 'outlook_iphone',
                            'android': 'outlook_android',
                            'thunderbird': 'outlook_thunderbird',
                            'outro': 'outlook_generic'
                        },
                        'fallback_message': "Em qual aplicativo quer configurar o Outlook/Hotmail?"
                    },
                    'corporate_setup': {
                        'message': "Email corporativo! 🏢\n\nPara configurar email da empresa, você precisa das informações do seu departamento de TI:\n\n• Servidor de entrada (IMAP/POP3)\n• Servidor de saída (SMTP)\n• Portas e segurança\n• Seu usuário e senha\n\nVocê tem essas informações?",
                        'expected_responses': {
                            'sim': 'corporate_manual_setup',
                            'yes': 'corporate_manual_setup',
                            'tenho': 'corporate_manual_setup',
                            'nao': 'contact_it_support',
                            'não': 'contact_it_support',
                            'no': 'contact_it_support'
                        },
                        'fallback_message': "Você tem as informações de configuração do email corporativo? Responda 'sim' ou 'não'."
                    },
                    'contact_it_support': {
                        'message': "Para email corporativo, você precisa entrar em contato com:\n• Departamento de TI da empresa\n• Suporte técnico interno\n• Administrador de sistemas\n\nEles fornecerão:\n• Configurações específicas\n• Usuário e senha\n• Instruções de segurança\n\nApós obter as informações, posso ajudar com a configuração!",
                        'expected_responses': {
                            'ok': 'wait_it_info',
                            'entendi': 'wait_it_info',
                            'vou contatar': 'wait_it_info'
                        },
                        'fallback_message': "Entre em contato com o TI da empresa para obter as configurações. Depois posso ajudar!"
                    },
                    'test_email_send': {
                        'message': "Ótimo! O email está configurado. Vamos testar:\n\n1. Compose um novo email\n2. Envie para você mesmo\n3. Verifique se recebe o email\n4. Teste responder\n\nO teste de envio e recebimento funcionou?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'recebeu': 'success',
                            'nao': 'email_test_troubleshoot',
                            'não': 'email_test_troubleshoot',
                            'no': 'email_test_troubleshoot'
                        },
                        'fallback_message': "O teste de envio e recebimento funcionou? Responda 'sim' ou 'não'."
                    },
                    'email_stopped': {
                        'message': "Email parou de funcionar. Vamos diagnosticar:\n\nO que acontece quando você tenta usar o email?\n• Pede senha constantemente\n• Erro de conexão\n• Não baixa emails novos\n• Não consegue enviar\n• Outro erro",
                        'expected_responses': {
                            'senha': 'password_problem',
                            'pede senha': 'password_problem',
                            'conexao': 'connection_error',
                            'erro conexao': 'connection_error',
                            'nao baixa': 'receive_problem',
                            'não baixa': 'receive_problem',
                            'nao envia': 'send_problem',
                            'não envia': 'send_problem',
                            'outro': 'other_email_problem'
                        },
                        'fallback_message': "O que acontece quando tenta usar o email? Pede senha, erro de conexão, não baixa, não envia ou outro problema?"
                    },
                    'password_problem': {
                        'message': "Problema de senha. Vamos resolver:\n\n1. Sua senha do email mudou recentemente?\n2. Você ativou autenticação em 2 fatores?\n3. O provedor mudou políticas de segurança?\n\nVamos atualizar a senha no aplicativo. Qual é seu provedor de email?",
                        'expected_responses': {
                            'gmail': 'gmail_password_update',
                            'outlook': 'outlook_password_update',
                            'yahoo': 'yahoo_password_update',
                            'corporativo': 'corporate_password_update'
                        },
                        'fallback_message': "Qual é seu provedor de email? Gmail, Outlook, Yahoo ou corporativo?"
                    },
                    'send_problem': {
                        'message': "Problema para enviar emails. Vamos verificar:\n\n1. Os emails ficam na caixa de saída?\n2. Recebe mensagem de erro específica?\n3. O problema é com todos os destinatários?\n4. Anexos muito grandes?\n\nO que você observa quando tenta enviar?",
                        'expected_responses': {
                            'caixa saida': 'outbox_problem',
                            'erro': 'send_error_analysis',
                            'todos': 'smtp_problem',
                            'anexos': 'attachment_size_problem',
                            'grandes': 'attachment_size_problem'
                        },
                        'fallback_message': "O que acontece quando tenta enviar? Fica na caixa de saída, dá erro, problema com todos ou anexos grandes?"
                    },
                    'receive_problem': {
                        'message': "Problema para receber emails. Vamos verificar:\n\n1. Há quanto tempo não recebe emails?\n2. A caixa de entrada está cheia?\n3. Emails vão para spam?\n4. Problema com remetentes específicos?\n\nHá quanto tempo não recebe emails novos?",
                        'expected_responses': {
                            'hoje': 'recent_receive_problem',
                            'ontem': 'recent_receive_problem',
                            'dias': 'old_receive_problem',
                            'semana': 'old_receive_problem',
                            'cheia': 'mailbox_full',
                            'spam': 'spam_problem'
                        },
                        'fallback_message': "Há quanto tempo não recebe emails? Hoje, dias, semana, ou a caixa está cheia?"
                    },
                    'escalate_support': {
                        'message': "O problema parece mais complexo. Recomendo:\n\n• Contatar suporte do provedor de email\n• Verificar configurações avançadas\n• Considerar reconfiguração completa\n• Backup dos emails importantes\n\nPosso ajudar com configuração básica, mas problemas complexos podem precisar de suporte especializado.",
                        'expected_responses': {},
                        'solution': "Caso escalado para suporte especializado do provedor."
                    },
                    'success': {
                        'message': "Fantástico! 🎉 Seu email está configurado e funcionando perfeitamente!\n\nDicas para manter o email funcionando:\n• Mantenha senhas atualizadas\n• Configure backup regular\n• Organize pastas e regras\n• Mantenha aplicativo atualizado\n\nPosso ajudar com mais alguma coisa?",
                        'expected_responses': {},
                        'solution': "Email configurado e funcionando com sucesso."
                    }
                }
            }
        }
    
    def start_interactive_flow(self, flow_name: str, user_id: str) -> str:
        """Inicia um fluxo interativo específico"""
        if flow_name not in self.interactive_flows:
            return f"Fluxo '{flow_name}' não encontrado."
        
        # Inicializar contexto do usuário
        self.conversation_context[user_id] = {
            'flow_name': flow_name,
            'current_step': 'start',
            'state': 'interactive_flow'
        }
        
        # Retornar primeira mensagem do fluxo
        flow = self.interactive_flows[flow_name]
        start_step = flow['steps']['start']
        return start_step['message']
    
    def process_interactive_response(self, user_response: str, user_id: str) -> str:
        """Processa resposta do usuário em um fluxo interativo"""
        if user_id not in self.conversation_context:
            return "Erro: Contexto de conversa não encontrado."
        
        context = self.conversation_context[user_id]
        flow_name = context['flow_name']
        current_step = context['current_step']
        
        if flow_name not in self.interactive_flows:
            return "Erro: Fluxo não encontrado."
        
        flow = self.interactive_flows[flow_name]
        step_data = flow['steps'][current_step]
        
        # Normalizar resposta do usuário
        user_response_lower = user_response.lower().strip()
        
        # Verificar respostas esperadas
        next_step = None
        for expected_response, next_step_id in step_data['expected_responses'].items():
            if expected_response in user_response_lower:
                next_step = next_step_id
                break
        
        # Se não encontrou correspondência, usar fallback
        if not next_step:
            return step_data.get('fallback_message', "Não entendi sua resposta. Pode tentar novamente?")
        
        # Atualizar contexto para próximo passo
        context['current_step'] = next_step
        
        # Verificar se chegou ao fim do fluxo
        if next_step not in flow['steps']:
            return f"Erro: Passo '{next_step}' não encontrado no fluxo."
        
        next_step_data = flow['steps'][next_step]
        
        # Se tem solução, significa que o fluxo terminou
        if 'solution' in next_step_data:
            # Mudar para estado pós-diagnóstico
            context['state'] = 'post_diagnostic'
            context['solution'] = next_step_data['solution']
            return next_step_data['message']
        
        # Retornar próxima mensagem
        return next_step_data['message']
    
    def start_diagnostic(self, diagnostic_type: str, user_id: str) -> str:
        """Inicia diagnóstico (compatibilidade com sistema antigo para Wi-Fi)"""
        # Mapear diagnósticos antigos para novos fluxos interativos
        flow_mapping = {
            'wifi_troubleshooting': 'wifi_troubleshooting',
            'password_recovery': 'password_recovery',
            'printer_troubleshooting': 'printer_troubleshooting',
            'email_configuration': 'email_configuration'
        }
        
        if diagnostic_type in flow_mapping:
            return self.start_interactive_flow(flow_mapping[diagnostic_type], user_id)
        
        return f"Diagnóstico '{diagnostic_type}' não disponível."
    
    def process_answer(self, answer: str, user_id: str) -> str:
        """Processa resposta (compatibilidade com sistema antigo)"""
        return self.process_interactive_response(answer, user_id)


class EnhancedChatbot:
    """Chatbot aprimorado com fluxos interativos para todas as categorias"""
    
    def __init__(self):
        self.nlp_engine = AdvancedNLPEngine()
        self.inference_engine = LogicalInferenceEngine()
        
        # Base de conhecimento expandida
        self.knowledge_base = {
            "wifi": {
                "keywords": ["wifi", "wi-fi", "internet", "rede", "conexao", "conectar", "wireless", "sem fio", "roteador", "modem", "router", "sinal", "banda larga"],
                "phrases": [
                    "como configurar wifi", "nao consigo conectar na internet", "wifi nao funciona",
                    "problema com internet", "rede sem fio", "configurar roteador", "senha do wifi",
                    "internet lenta", "sinal fraco", "nao conecta no wifi", "wifi desconectando"
                ],
                "confidence_threshold": 1.5
            },
            "senha": {
                "keywords": ["senha", "password", "esqueci", "resetar", "redefinir", "recuperar", "login", "acesso", "credencial"],
                "phrases": [
                    "esqueci minha senha", "como resetar senha", "recuperar senha", "nao lembro a senha",
                    "perdi minha senha", "redefinir password", "problema de login", "nao consigo entrar",
                    "senha incorreta", "bloqueado por senha", "alterar senha"
                ],
                "confidence_threshold": 1.5
            },
            "impressora": {
                "keywords": ["impressora", "imprimir", "printer", "papel", "tinta", "cartucho", "toner", "scanner", "impressao"],
                "phrases": [
                    "impressora nao funciona", "nao consigo imprimir", "problema na impressora",
                    "impressora offline", "papel atolado", "tinta acabou", "cartucho vazio",
                    "impressora nao conecta", "erro de impressao", "impressora lenta"
                ],
                "confidence_threshold": 1.5
            },
            "email": {
                "keywords": ["email", "e-mail", "outlook", "gmail", "correio", "mail", "yahoo", "hotmail", "mensagem"],
                "phrases": [
                    "como configurar email", "problema no email", "nao recebo emails",
                    "email nao funciona", "configurar outlook", "gmail nao abre",
                    "problema com correio", "nao consigo enviar email", "email travando"
                ],
                "confidence_threshold": 1.5
            }
        }
        
        self.greetings = ["ola", "oi", "bom dia", "boa tarde", "boa noite", "hello", "hi", "opa"]
        self.farewells = ["tchau", "ate logo", "obrigado", "valeu", "bye", "flw", "vlw"]
    
    def classify_intent_advanced(self, text: str) -> Tuple[str, float]:
        """Classificação avançada de intenção com score de confiança"""
        tokens = self.nlp_engine.preprocess_text(text)
        text_lower = text.lower()
        
        # Verificar saudações
        for greeting in self.greetings:
            if greeting in tokens:
                return "greeting", 3.0
        
        # Verificar despedidas
        for farewell in self.farewells:
            if farewell in tokens:
                return "farewell", 3.0
        
        # Verificar diagnósticos
        if "diagnostico" in tokens:
            for topic in self.knowledge_base.keys():
                if topic in tokens:
                    return f"diagnostic_{topic}", 3.0
        
        # Classificação por similaridade
        best_intent = None
        best_score = 0.0
        
        for topic, data in self.knowledge_base.items():
            # Verificar correspondência com frases completas
            phrase_score = 0.0
            for phrase in data.get("phrases", []):
                phrase_tokens = self.nlp_engine.preprocess_text(phrase)
                similarity = self.nlp_engine.calculate_similarity(tokens, phrase_tokens)
                phrase_score = max(phrase_score, similarity)
            
            # Verificar correspondência com palavras-chave
            keyword_tokens = self.nlp_engine.preprocess_text(" ".join(data["keywords"]))
            keyword_similarity = self.nlp_engine.calculate_similarity(tokens, keyword_tokens)
            
            # Usar a maior pontuação
            final_score = max(phrase_score, keyword_similarity)
            
            if final_score > best_score and final_score >= data.get("confidence_threshold", 1.0):
                best_score = final_score
                best_intent = topic
        
        return (best_intent, best_score) if best_intent else ("unknown", 0.0)
    
    def process_message(self, message: str, user_id: str = "default") -> Dict:
        """Processa uma mensagem com IA avançada e fluxos interativos"""
        message_lower = message.lower().strip()
        
        # Verificar se há contexto de conversa ativo
        if user_id in self.inference_engine.conversation_context:
            context = self.inference_engine.conversation_context[user_id]
            
            # Se está em fluxo interativo
            if context.get('state') == 'interactive_flow':
                response = self.inference_engine.process_interactive_response(message, user_id)
                return {
                    "response": response,
                    "type": "interactive_flow",
                    "confidence": 3.0
                }
            
            # Se estamos no estado pós-diagnóstico
            elif context.get('state') == 'post_diagnostic':
                # Verificar se o usuário quer mais ajuda
                if any(word in message_lower for word in ['sim', 'yes', 'claro', 'quero', 'preciso', 'gostaria']):
                    # Limpar contexto e voltar ao estado inicial
                    del self.inference_engine.conversation_context[user_id]
                    return {
                        "response": "Ótimo! Como posso ajudá-lo agora? Posso auxiliar com:\n\n• Configuração de Wi-Fi\n• Redefinição de senhas\n• Problemas com impressora\n• Configuração de email\n\nOu digite 'diagnóstico' seguido do problema para um atendimento personalizado.",
                        "type": "menu",
                        "confidence": 3.0
                    }
                elif any(word in message_lower for word in ['nao', 'no', 'obrigado', 'tchau', 'ate logo', 'valeu']):
                    # Encerrar atendimento
                    del self.inference_engine.conversation_context[user_id]
                    return {
                        "response": "Foi um prazer ajudá-lo! Se precisar de mais alguma coisa, estarei aqui. Tenha um ótimo dia! 😊",
                        "type": "farewell",
                        "confidence": 3.0
                    }
                else:
                    # Não entendeu a resposta, perguntar novamente
                    return {
                        "response": "Desculpe, não entendi. Você gostaria de mais ajuda? Responda 'sim' para continuar ou 'não' para encerrar o atendimento.",
                        "type": "clarification",
                        "confidence": 3.0
                    }
        
        # Classificação normal de intenção
        intent, confidence = self.classify_intent_advanced(message)
        
        # Verificar se é comando para iniciar fluxo interativo
        if intent in ['wifi', 'senha', 'impressora', 'email'] or intent.startswith('diagnostic_'):
            # Mapear intenções para fluxos
            flow_mapping = {
                'wifi': 'wifi_troubleshooting',
                'senha': 'password_recovery', 
                'impressora': 'printer_troubleshooting',
                'email': 'email_configuration',
                'diagnostic_wifi': 'wifi_troubleshooting',
                'diagnostic_senha': 'password_recovery',
                'diagnostic_impressora': 'printer_troubleshooting',
                'diagnostic_email': 'email_configuration'
            }
            
            flow_name = flow_mapping.get(intent)
            if flow_name:
                response = self.inference_engine.start_interactive_flow(flow_name, user_id)
                return {
                    "response": response,
                    "type": "interactive_flow_start",
                    "confidence": 3.0
                }
        
        # Respostas para outros tipos de intenção
        if intent == "greeting":
            return {
                "response": "Olá! Sou seu assistente de suporte técnico com IA avançada. Como posso ajudá-lo hoje?\n\nPosso auxiliar com:\n• Configuração de Wi-Fi\n• Redefinição de senhas\n• Problemas com impressora\n• Configuração de email",
                "type": "greeting",
                "confidence": confidence
            }
        
        elif intent == "farewell":
            return {
                "response": "Obrigado por usar nosso suporte! Tenha um ótimo dia!",
                "type": "farewell", 
                "confidence": confidence
            }
        
        else:
            # Fallback inteligente
            tokens = self.nlp_engine.preprocess_text(message)
            recognized_topics = []
            
            for topic, data in self.knowledge_base.items():
                for keyword in data["keywords"]:
                    if keyword in tokens:
                        recognized_topics.append(topic)
                        break
            
            if recognized_topics:
                topics_text = ", ".join(recognized_topics)
                fallback_response = f"Não entendi completamente, mas notei que você mencionou algo sobre {topics_text}. Você está com problemas relacionados a isso?\n\nPosso ajudar com: configuração de Wi-Fi, redefinição de senhas, problemas com impressora ou configuração de email."
            else:
                fallback_response = f"Desculpe, não consegui entender sua pergunta com confiança suficiente (confiança: {confidence:.1f}).\n\nPosso ajudar com:\n• Configuração de Wi-Fi\n• Redefinição de senhas\n• Problemas com impressora\n• Configuração de email\n\nPode reformular sua pergunta ou ser mais específico?"
            
            return {
                "response": fallback_response,
                "type": "unknown",
                "confidence": confidence
            }

