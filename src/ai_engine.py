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
                if intent in ['senha', 'impressora', 'email']:
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
                        'message': "Olá! Vejo que você selecionou a opção 'Senha'. Para te ajudar a redefinir sua senha, por favor, acesse a página de login do sistema. Conseguiu chegar lá?",
                        'expected_responses': {
                            'sim': 'click_forgot',
                            'yes': 'click_forgot',
                            'consegui': 'click_forgot',
                            'ok': 'click_forgot',
                            'nao': 'help_find_login',
                            'não': 'help_find_login',
                            'no': 'help_find_login',
                            'nao sei': 'help_find_login'
                        }
                    },
                    'help_find_login': {
                        'message': "Sem problemas! Para encontrar a página de login:\n1. Abra seu navegador.\n2. Digite o endereço do site ou sistema.\n3. Procure por 'Login', 'Entrar' ou 'Acesso'.\n\nMe diga quando conseguir. Conseguiu encontrar agora?",
                        'expected_responses': {
                            'sim': 'click_forgot',
                            'yes': 'click_forgot',
                            'consegui': 'click_forgot',
                            'ok': 'click_forgot',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        }
                    },
                    'click_forgot': {
                        'message': "Ótimo! Agora na página de login, clique em 'Esqueci minha senha'. Você está vendo essa opção?",
                        'expected_responses': {
                            'sim': 'enter_email',
                            'yes': 'enter_email',
                            'vejo': 'enter_email',
                            'cliquei': 'enter_email',
                            'ja cliquei': 'enter_email',
                            'nao': 'help_find_forgot',
                            'não': 'help_find_forgot',
                            'no': 'help_find_forgot'
                        }
                    },
                    'help_find_forgot': {
                        'message': "A opção pode estar com nomes como 'Recuperar senha', 'Redefinir password' ou 'Forgot password'. Geralmente fica abaixo dos campos de login. Encontrou?",
                        'expected_responses': {
                            'sim': 'enter_email',
                            'yes': 'enter_email',
                            'encontrei': 'enter_email',
                            'achei': 'enter_email',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        }
                    },
                    'enter_email': {
                        'message': "Perfeito. Agora digite o seu e-mail cadastrado e clique em 'enviar' ou 'continuar'. Você já fez isso?",
                        'expected_responses': {
                            'sim': 'check_email',
                            'yes': 'check_email',
                            'pronto': 'check_email',
                            'feito': 'check_email',
                            'ja fiz': 'check_email',
                            'digitei': 'check_email',
                            'enviei': 'check_email'
                        }
                    },
                    'check_email': {
                        'message': "Beleza! Em alguns instantes você deve receber um e-mail com um link para redefinir sua senha. Pode verificar na sua caixa de entrada, ou na pasta de spam, caso não apareça logo? Me diga se o e-mail chegou.",
                        'expected_responses': {
                            'sim': 'click_link',
                            'yes': 'click_link',
                            'recebi': 'click_link',
                            'chegou': 'click_link',
                            'nao': 'email_troubleshoot',
                            'não': 'email_troubleshoot',
                            'no': 'email_troubleshoot',
                            'nao chegou': 'email_troubleshoot'
                        }
                    },
                    'email_troubleshoot': {
                        'message': "Sem problemas! Vamos verificar algumas coisas:\n1. Confira a pasta de spam/lixo eletrônico.\n2. Aguarde mais alguns minutos (pode demorar até 10 min).\n3. Verifique se digitou o e-mail correto.\n\nO e-mail chegou agora?",
                        'expected_responses': {
                            'sim': 'click_link',
                            'yes': 'click_link',
                            'recebi': 'click_link',
                            'chegou': 'click_link',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        }
                    },
                    'click_link': {
                        'message': "Maravilha! Agora clique no link que você recebeu e siga as instruções para escolher uma nova senha. Depois de redefinir, me avise.",
                        'expected_responses': {
                            'sim': 'test_login',
                            'yes': 'test_login',
                            'pronto': 'test_login',
                            'feito': 'test_login',
                            'redefinida': 'test_login',
                            'alterada': 'test_login'
                        }
                    },
                    'test_login': {
                        'message': "Perfeito! Agora tente fazer login novamente com a nova senha. Conseguiu acessar sua conta?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'consegui': 'success',
                            'funcionou': 'success',
                            'entrei': 'success',
                            'nao': 'login_troubleshoot',
                            'não': 'login_troubleshoot',
                            'no': 'login_troubleshoot'
                        }
                    },
                    'login_troubleshoot': {
                        'message': "Vamos verificar:\n1. Certifique-se de que está digitando a senha correta.\n2. Verifique se a tecla Caps Lock não está ativada.\n3. Tente copiar e colar a senha do e-mail de redefinição.\n\nFuncionou agora?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support'
                        }
                    },
                    'escalate_support': {
                        'message': "Entendo que precisa de uma ajuda mais específica. Vou te conectar com um atendente humano que poderá ajudar melhor com seu caso. Enquanto isso, você pode tentar reiniciar o seu navegador. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'success': {
                        'message': "Excelente! Sua senha foi redefinida com sucesso e você conseguiu acessar sua conta. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'flow_continue': {
                        'message': "Ótimo! Posso ajudar com:\n\n- Problemas com impressora\n- Configuração de e-mail",
                        'expected_responses': {
                            'impressora': 'printer_troubleshooting',
                            'email': 'email_configuration',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                        },
                        'solution': "Oferecendo menu principal."
                    },
                    'flow_exit': {
                        'message': "Foi um prazer ajudar! Tenha um ótimo dia. 😊",
                        'expected_responses': {},
                        'solution': "O usuário optou por sair do fluxo de recuperação de senha."
                    }
                }
            },
            'printer_troubleshooting': {
                'steps': {
                    'start': {
                        'message': "Olá! Vejo que você está com problemas na impressora. Vamos resolver isso. Primeiro, me diga: qual é o problema específico? Responda com a opção que melhor descreve o seu problema:\n\n- **Não imprime**\n- **Papel atolado**\n- **Qualidade ruim**\n- **Não reconhece a impressora**\n- **Outro problema**",
                        'expected_responses': {
                            'nao imprime': 'check_power',
                            'não imprime': 'check_power',
                            'nada': 'check_power',
                            'papel': 'paper_jam',
                            'atolado': 'paper_jam',
                            'qualidade': 'print_quality',
                            'ruim': 'print_quality',
                            'nao reconhece': 'connection_issue',
                            'não reconhece': 'connection_issue',
                            'reconhece': 'connection_issue',
                            'outro': 'other_printer_problem',
                        }
                    },
                    'check_power': {
                        'message': "Vamos verificar o básico. A impressora está ligada? As luzes estão acesas?",
                        'expected_responses': {
                            'sim': 'check_connection',
                            'yes': 'check_connection',
                            'ligada': 'check_connection',
                            'luzes': 'check_connection',
                            'nao': 'power_troubleshoot',
                            'não': 'power_troubleshoot',
                            'no': 'power_troubleshoot',
                            'desligada': 'power_troubleshoot',
                        }
                    },
                    'power_troubleshoot': {
                        'message': "Vamos ligar a impressora. Verifique se o cabo de energia está bem conectado e se a tomada está funcionando. Conseguiu ligar?",
                        'expected_responses': {
                            'sim': 'check_connection',
                            'yes': 'check_connection',
                            'ligou': 'check_connection',
                            'funcionou': 'check_connection',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'check_connection': {
                        'message': "Ótimo! A impressora está ligada. Como ela está conectada ao seu computador? Responda com a opção:\n\n- **Cabo USB**\n- **Wi-Fi**\n- **Cabo de rede** (Ethernet)\n- **Bluetooth**",
                        'expected_responses': {
                            'usb': 'check_usb',
                            'cabo': 'check_usb',
                            'wifi': 'check_wifi_printer',
                            'wi-fi': 'check_wifi_printer',
                            'rede': 'check_ethernet',
                            'ethernet': 'check_ethernet',
                            'bluetooth': 'check_bluetooth',
                        }
                    },
                    'check_usb': {
                        'message': "Conexão via USB. O cabo está bem conectado nas duas pontas? Tente conectar em outra porta USB do seu computador. O computador reconhece a impressora?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'reconhece': 'test_print',
                            'mostra': 'test_print',
                            'nao': 'usb_troubleshoot',
                            'não': 'usb_troubleshoot',
                            'no': 'usb_troubleshoot',
                        }
                    },
                    'usb_troubleshoot': {
                        'message': "Vamos resolver a conexão USB. Tente testar outro cabo USB, se tiver. E, se o problema continuar, pode ser que você precise instalar os drivers da impressora. Quer que eu te ajude a encontrar os drivers?",
                        'expected_responses': {
                            'sim': 'driver_install',
                            'yes': 'driver_install',
                            'quero': 'driver_install',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'check_wifi_printer': {
                        'message': "Impressora Wi-Fi. A impressora está conectada na mesma rede Wi-Fi que o seu computador?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'mesma': 'test_print',
                            'rede': 'test_print',
                            'nao': 'wifi_printer_setup',
                            'não': 'wifi_printer_setup',
                            'no': 'wifi_printer_setup',
                        }
                    },
                    'wifi_printer_setup': {
                        'message': "Vamos conectar a impressora ao Wi-Fi. No painel da impressora, procure por 'Configurações' ou 'Wi-Fi'. Conseguiu encontrar?",
                        'expected_responses': {
                            'sim': 'wifi_connect_printer',
                            'yes': 'wifi_connect_printer',
                            'encontrei': 'wifi_connect_printer',
                            'achei': 'wifi_connect_printer',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'wifi_connect_printer': {
                        'message': "Ótimo! Agora selecione sua rede Wi-Fi na lista e digite a senha. A impressora conectou? (Geralmente aparece um ícone ou mensagem de confirmação).",
                        'expected_responses': {
                            'sim': 'add_printer_computer',
                            'yes': 'add_printer_computer',
                            'conectou': 'add_printer_computer',
                            'funcionou': 'add_printer_computer',
                            'nao': 'wifi_password_help',
                            'não': 'wifi_password_help',
                            'no': 'wifi_password_help',
                        }
                    },
                    'wifi_password_help': {
                        'message': "Pode ser um erro na senha ou na rede. Verifique se a senha está correta e se a rede é 2.4GHz. Tente novamente. Funcionou?",
                        'expected_responses': {
                            'sim': 'add_printer_computer',
                            'yes': 'add_printer_computer',
                            'funcionou': 'add_printer_computer',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'add_printer_computer': {
                        'message': "Perfeito! A impressora está no Wi-Fi. Agora vamos adicioná-la ao computador. Em 'Configurações' ou 'Preferências', procure por 'Impressoras e scanners' e adicione a sua impressora. Conseguiu adicionar?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'adicionei': 'test_print',
                            'funcionou': 'test_print',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'check_ethernet': {
                        'message': "Conexão por cabo de rede. Verifique se o cabo está bem conectado tanto na impressora quanto no roteador ou na parede. A impressora tem alguma luz indicando a conexão de rede? Ela está acesa?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'acendeu': 'test_print',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'check_bluetooth': {
                        'message': "Conexão Bluetooth. Certifique-se de que o Bluetooth da impressora e do seu computador estão ligados e pareados. A impressora aparece na lista de dispositivos Bluetooth no seu computador?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'aparece': 'test_print',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'connection_issue': {
                        'message': "A impressora não está sendo reconhecida. Vamos verificar a conexão. Como ela está conectada ao seu computador? Responda com a opção:\n\n- **Cabo USB**\n- **Wi-Fi**\n- **Cabo de rede** (Ethernet)\n- **Bluetooth**",
                        'expected_responses': {
                            'usb': 'check_usb',
                            'cabo': 'check_usb',
                            'wifi': 'check_wifi_printer',
                            'wi-fi': 'check_wifi_printer',
                            'rede': 'check_ethernet',
                            'ethernet': 'check_ethernet',
                            'bluetooth': 'check_bluetooth',
                        }
                    },
                    'driver_install': {
                        'message': "Vamos instalar os drivers. Você precisa acessar o site do fabricante (HP, Canon, Epson, etc.), procurar pelo modelo da sua impressora e baixar os drivers. Você conseguiu fazer isso?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'baixei': 'test_print',
                            'instalei': 'test_print',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'test_print': {
                        'message': "Excelente! Agora vamos testar. Tente imprimir uma página de teste ou um documento simples. A impressora imprimiu?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'imprimiu': 'success',
                            'funcionou': 'success',
                            'nao': 'print_troubleshoot',
                            'não': 'print_troubleshoot',
                            'no': 'print_troubleshoot',
                        }
                    },
                    'print_troubleshoot': {
                        'message': "O que você observa? Há alguma mensagem de erro ou luz piscando na impressora? Responda com a opção que melhor descreve o seu problema:\n\n- **Papel atolado**\n- **Falta tinta ou toner**\n- **Outro erro**",
                        'expected_responses': {
                            'papel': 'paper_jam',
                            'tinta': 'ink_issue',
                            'toner': 'ink_issue',
                            'outro': 'escalate_support',
                        }
                    },
                    'paper_jam': {
                        'message': "Para resolver o papel atolado:\n1. Desligue a impressora.\n2. Abra todas as tampas.\n3. Remova cuidadosamente o papel atolado, sem forçar.\n4. Feche as tampas e ligue a impressora.\n\nConseguiu remover todo o papel?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'removi': 'test_print',
                            'limpei': 'test_print',
                            'nao': 'paper_jam_help',
                            'não': 'paper_jam_help',
                            'no': 'paper_jam_help',
                        }
                    },
                    'paper_jam_help': {
                        'message': "Se o papel está difícil de remover, não force! Isso pode danificar a impressora. Recomendo consultar o manual da sua impressora para ver a forma correta de remover o papel atolado. Conseguiu resolver?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'consegui': 'test_print',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'ink_issue': {
                        'message': "Vamos verificar a tinta ou toner. Remova o cartucho, verifique se ele tem tinta e recoloque-o firmemente. Se estiver vazio, substitua-o. Fez isso? A impressora agora imprime?",
                        'expected_responses': {
                            'sim': 'test_print',
                            'yes': 'test_print',
                            'troquei': 'test_print',
                            'instalei': 'test_print',
                            'nao': 'ink_issue_help',
                            'não': 'ink_issue_help',
                            'no': 'ink_issue_help',
                        }
                    },
                    'ink_issue_help': {
                        'message': "Se o cartucho ainda tem tinta, mas a impressão está falhando, pode ser que as saídas estejam entupidas. Em 'Manutenção' nas configurações da impressora, procure por uma opção de 'Limpeza Profunda' ou 'Alinhamento de Cartuchos'. Tente isso. A impressão melhorou?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'melhorou': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'print_quality': {
                        'message': "Problema de qualidade de impressão. Qual o problema? Responda com a opção que melhor descreve o seu problema:\n\n- **Borrado ou manchado**\n- **Cores desbotadas**\n- **Linhas ou riscos**\n- **Outro**",
                        'expected_responses': {
                            'borrado': 'clean_heads',
                            'manchado': 'clean_heads',
                            'cores': 'clean_heads',
                            'desbotadas': 'clean_heads',
                            'linhas': 'clean_heads',
                            'riscos': 'clean_heads',
                            'outro': 'escalate_support',
                        }
                    },
                    'clean_heads': {
                        'message': "Vamos tentar limpar os cabeçotes de impressão. Nas configurações da impressora no seu computador, procure por 'Manutenção' ou 'Limpeza'. Execute o processo e depois imprima uma página de teste. A qualidade melhorou?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'melhorou': 'success',
                            'nao': 'clean_heads_help',
                            'não': 'clean_heads_help',
                            'no': 'clean_heads_help',
                        }
                    },
                    'clean_heads_help': {
                        'message': "Se a limpeza padrão não resolveu, tente a 'Limpeza Profunda' ou 'Deep Cleaning'. Se mesmo assim não funcionar, o problema pode ser físico com os cartuchos ou a impressora. Conseguiu resolver?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'consegui': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'other_printer_problem': {
                        'message': "Pode me descrever melhor o problema? Por exemplo, 'A impressora está muito lenta' ou 'Não consigo digitalizar'.",
                        'expected_responses': {
                            'lenta': 'escalate_support',
                            'digitalizar': 'escalate_support',
                            'nao sei': 'escalate_support'
                        }
                    },
                    'escalate_support': {
                        'message': "O problema parece mais complexo. Vou te conectar com um técnico especializado. Enquanto aguarda, anote o modelo do seu roteador e verifique se há alguma luz piscando. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'success': {
                        'message': "Excelente! Sua impressora está funcionando perfeitamente! Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'flow_continue': {
                        'message': "Ótimo! Posso ajudar com:\n\n- Redefinição de senhas\n- Problemas com impressora\n- Configuração de e-mail",
                        'expected_responses': {
                            'senha': 'password_recovery',
                            'impressora': 'printer_troubleshooting',
                            'email': 'email_configuration',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                        },
                        'solution': "Oferecendo menu principal."
                    },
                    'flow_exit': {
                        'message': "Foi um prazer ajudar! Tenha um ótimo dia. 😊",
                        'expected_responses': {},
                        'solution': "O usuário optou por sair do fluxo da impressora."
                    }
                }
            },
            'email_configuration': {
                'steps': {
                    'start': {
                        'message': "Olá! Vejo que você precisa de ajuda com configuração de e-mail. Qual é a sua situação? Responda com a opção que melhor descreve o seu problema:\n\n- **Configurar pela primeira vez**\n- **Não consigo enviar e-mails**\n- **Não consigo receber e-mails**\n- **E-mail parou de funcionar**\n- **Outro problema**",
                        'expected_responses': {
                            'primeira vez': 'first_time_setup',
                            'primeira': 'first_time_setup',
                            'configurar': 'first_time_setup',
                            'enviar': 'send_problem',
                            'nao consigo enviar': 'send_problem',
                            'receber': 'receive_problem',
                            'nao recebo': 'receive_problem',
                            'parou': 'email_stopped',
                            'nao funciona': 'email_stopped',
                            'outro': 'other_email_problem',
                        }
                    },
                    'first_time_setup': {
                        'message': "Vamos configurar seu e-mail! Qual provedor você usa? Responda com o nome do seu provedor:\n\n- **Gmail**\n- **Outlook** (Hotmail/Live)\n- **Yahoo**\n- **Corporativo** (da empresa)\n- **Outro provedor**",
                        'expected_responses': {
                            'gmail': 'gmail_setup',
                            'google': 'gmail_setup',
                            'outlook': 'outlook_setup',
                            'hotmail': 'outlook_setup',
                            'yahoo': 'yahoo_setup',
                            'corporativo': 'corporate_setup',
                            'trabalho': 'corporate_setup',
                            'empresa': 'corporate_setup',
                            'outro': 'escalate_support',
                        }
                    },
                    'gmail_setup': {
                        'message': "Configuração do Gmail. Na maioria dos aplicativos de e-mail, basta digitar seu endereço e senha, e as configurações são automáticas. Você já tentou isso?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'ja fiz': 'test_email_send',
                            'nao': 'gmail_manual_setup_offer',
                            'não': 'gmail_manual_setup_offer',
                            'no': 'gmail_manual_setup_offer',
                        }
                    },
                    'gmail_manual_setup_offer': {
                        'message': "Se a configuração automática falhou, pode ser um problema de autenticação. Para resolver, você pode precisar de uma 'Senha de App'. Quer tentar gerar uma?",
                        'expected_responses': {
                            'sim': 'gmail_app_password',
                            'yes': 'gmail_app_password',
                            'quero': 'gmail_app_password',
                            'gerar': 'gmail_app_password',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'gmail_app_password': {
                        'message': "Gerando senha de app:\n1. Acesse **myaccount.google.com**.\n2. Vá em 'Segurança' > 'Verificação em duas etapas' > 'Senhas de app'.\n3. Gere uma nova senha para o seu aplicativo de e-mail.\n4. Use essa senha gerada (não sua senha normal) no aplicativo.\n\nConseguiu gerar e usar a senha de app?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'funcionou': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'outlook_setup': {
                        'message': "Configuração do Outlook. Na maioria dos casos, o aplicativo detecta as configurações automaticamente. Apenas digite seu e-mail e senha. Conseguiu adicionar a conta?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'outlook_manual_setup',
                            'não': 'outlook_manual_setup',
                            'no': 'outlook_manual_setup',
                        }
                    },
                    'outlook_manual_setup': {
                        'message': "Se a configuração automática falhou, você precisará das configurações manuais. Recomendo procurar por 'Configurações de servidor Outlook' online. Quando tiver as informações, me diga que eu te ajudo. Você conseguiu encontrar?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'yahoo_setup': {
                        'message': "Configuração do Yahoo. Similar ao Gmail, você pode precisar de uma 'Senha de App' se a verificação em duas etapas estiver ativada. Você já gerou uma?",
                        'expected_responses': {
                            'sim': 'yahoo_app_password_instructions',
                            'yes': 'yahoo_app_password_instructions',
                            'nao': 'yahoo_app_password_instructions',
                            'não': 'yahoo_app_password_instructions',
                            'no': 'yahoo_app_password_instructions',
                        }
                    },
                    'yahoo_app_password_instructions': {
                        'message': "Para gerar a senha de app do Yahoo, acesse as configurações de segurança da sua conta e procure por 'Senhas de app'. Use essa senha no seu aplicativo de e-mail. Conseguiu?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'corporate_setup': {
                        'message': "Para e-mail corporativo, você precisa das informações do seu departamento de TI, como o servidor de entrada (IMAP/POP) e o servidor de saída (SMTP). Você tem essas informações?",
                        'expected_responses': {
                            'sim': 'corporate_manual_setup',
                            'yes': 'corporate_manual_setup',
                            'tenho': 'corporate_manual_setup',
                            'nao': 'contact_it_support',
                            'não': 'contact_it_support',
                            'no': 'contact_it_support',
                        }
                    },
                    'corporate_manual_setup': {
                        'message': "Com as informações em mãos, procure por 'Adicionar Conta Manualmente' no seu aplicativo de e-mail e insira os dados fornecidos pelo TI. Funcionou?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'contact_it_support': {
                        'message': "Para problemas com e-mail corporativo, você precisa entrar em contato com o suporte de TI da sua empresa. Eles podem fornecer as configurações corretas ou resolver o problema de forma mais específica. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'email_stopped': {
                        'message': "Seu e-mail parou de funcionar. O que acontece quando você tenta usá-lo? Responda com a opção que melhor descreve o seu problema:\n\n- **Pede senha** constantemente\n- **Erro de conexão**\n- **Não baixa e-mails novos**\n- **Não consigo enviar**",
                        'expected_responses': {
                            'senha': 'password_problem',
                            'pede senha': 'password_problem',
                            'conexao': 'escalate_support', 
                            'erro conexao': 'escalate_support', 
                            'nao baixa': 'receive_problem',
                            'não baixa': 'receive_problem',
                            'nao envia': 'send_problem',
                            'não envia': 'send_problem',
                        }
                    },
                    'password_problem': {
                        'message': "Se o aplicativo de e-mail pede a senha, provavelmente ela mudou. Qual é seu provedor de e-mail? (Gmail, Outlook, etc.)",
                        'expected_responses': {
                            'gmail': 'gmail_password_update',
                            'outlook': 'outlook_password_update',
                            'yahoo': 'yahoo_password_update',
                            'corporativo': 'corporate_password_update',
                        }
                    },
                    'gmail_password_update': {
                        'message': "Para atualizar a senha do Gmail, você pode precisar de uma 'Senha de App'. Vá nas configurações de segurança da sua conta Google para gerar uma e use-a no aplicativo. Funcionou?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'outlook_password_update': {
                        'message': "Para o Outlook, tente simplesmente atualizar a senha nas configurações da sua conta no aplicativo de e-mail. Se não funcionar, tente remover a conta e adicioná-la novamente. Funcionou?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'corporate_password_update': {
                        'message': "Para e-mail corporativo, é essencial entrar em contato com o suporte de TI da sua empresa para redefinir a senha e garantir que não haja bloqueios de segurança. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'send_problem': {
                        'message': "Problema para enviar e-mails. Verifique se os e-mails ficam presos na sua caixa de saída. Se sim, o problema é com o servidor de saída (SMTP). Tente reiniciar o aplicativo de e-mail. Funcionou?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'send_problem_help',
                            'não': 'send_problem_help',
                            'no': 'send_problem_help',
                        }
                    },
                    'send_problem_help': {
                        'message': "O servidor de saída (SMTP) pode estar com problemas. Verifique as configurações de porta e segurança com o seu provedor de e-mail. Se o problema persistir, pode ser um bloqueio de firewall ou do provedor. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'receive_problem': {
                        'message': "Problema para receber e-mails. Verifique se a sua caixa de entrada não está cheia ou se os e-mails não estão indo para a pasta de spam. Tente reiniciar o aplicativo. Funcionou?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'nao': 'receive_problem_help',
                            'não': 'receive_problem_help',
                            'no': 'receive_problem_help',
                        }
                    },
                    'receive_problem_help': {
                        'message': "Se a sua caixa de entrada não está cheia, o problema pode estar no servidor de entrada (IMAP/POP). Tente verificar as configurações de porta e segurança. Posso te ajudar a encontrar as configurações do seu provedor se for um serviço popular. Quer tentar?",
                        'expected_responses': {
                            'sim': 'find_provider_settings',
                            'yes': 'find_provider_settings',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'find_provider_settings': {
                        'message': "Qual é o seu provedor de e-mail? (Gmail, Outlook, Yahoo, etc.)",
                        'expected_responses': {
                            'gmail': 'gmail_settings_info',
                            'outlook': 'outlook_settings_info',
                            'yahoo': 'yahoo_settings_info',
                            'nao sei': 'escalate_support',
                            'outro': 'escalate_support'
                        }
                    },
                    'gmail_settings_info': {
                        'message': "As configurações para Gmail geralmente são:\n- **Servidor de entrada (IMAP)**: imap.gmail.com (Porta 993, SSL)\n- **Servidor de saída (SMTP)**: smtp.gmail.com (Porta 465 ou 587, SSL/TLS)\n\nVocê pode tentar inserir esses dados manualmente no seu aplicativo. Conseguiu?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'outlook_settings_info': {
                        'message': "As configurações para Outlook geralmente são:\n- **Servidor de entrada (IMAP)**: outlook.office365.com (Porta 993, SSL)\n- **Servidor de saída (SMTP)**: smtp.office365.com (Porta 587, TLS)\n\nVocê pode tentar inserir esses dados manualmente no seu aplicativo. Conseguiu?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'yahoo_settings_info': {
                        'message': "As configurações para Yahoo geralmente são:\n- **Servidor de entrada (IMAP)**: imap.mail.yahoo.com (Porta 993, SSL)\n- **Servidor de saída (SMTP)**: smtp.mail.yahoo.com (Porta 465, SSL)\n\nVocê pode tentar inserir esses dados manualmente no seu aplicativo. Conseguiu?",
                        'expected_responses': {
                            'sim': 'test_email_send',
                            'yes': 'test_email_send',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'other_email_problem': {
                        'message': "Pode me descrever melhor o problema? Por exemplo, 'Meu e-mail está travando' ou 'Não consigo abrir anexos'.",
                        'expected_responses': {
                            'travando': 'escalate_support',
                            'anexos': 'escalate_support',
                            'nao sei': 'escalate_support'
                        }
                    },
                    'test_email_send': {
                        'message': "Ótimo! Agora vamos testar. Envie um e-mail para você mesmo. Você conseguiu enviar e receber a mensagem?",
                        'expected_responses': {
                            'sim': 'success',
                            'yes': 'success',
                            'funcionou': 'success',
                            'recebeu': 'success',
                            'nao': 'escalate_support',
                            'não': 'escalate_support',
                            'no': 'escalate_support',
                        }
                    },
                    'escalate_support': {
                        'message': "O problema parece mais complexo. Recomendo entrar em contato com o suporte do seu provedor de e-mail. Eles poderão te ajudar com configurações mais avançadas. Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'success': {
                        'message': "Excelente! Seu e-mail está configurado e funcionando perfeitamente! Posso te ajudar com mais alguma coisa?",
                        'expected_responses': {
                            'sim': 'flow_continue',
                            'yes': 'flow_continue',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                            'no': 'flow_exit'
                        }
                    },
                    'flow_continue': {
                        'message': "Ótimo! Posso ajudar com:\n\n- Redefinição de senhas\n- Problemas com impressora\n- Configuração de e-mail",
                        'expected_responses': {
                            'senha': 'password_recovery',
                            'impressora': 'printer_troubleshooting',
                            'email': 'email_configuration',
                            'nao': 'flow_exit',
                            'não': 'flow_exit',
                        },
                        'solution': "Oferecendo menu principal."
                    },
                    'flow_exit': {
                        'message': "Foi um prazer ajudar! Tenha um ótimo dia. 😊",
                        'expected_responses': {},
                        'solution': "O usuário optou por sair do fluxo de e-mail."
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
            'state': 'interactive_flow',
            'fallback_count': 0
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
            context['fallback_count'] += 1
            if context['fallback_count'] >= 2:
                # Se 2 ou mais tentativas falharam, oferece ajuda
                context['fallback_count'] = 0
                return "Parece que não estou entendendo. Por favor, tente responder com as opções que eu te dei ou me diga 'não sei' para que eu possa te ajudar de outra forma."
            else:
                formatted_options = "\n- " + "\n- ".join(step_data['expected_responses'].keys())
                return f"Desculpe, não entendi. Por favor, tente responder com as seguintes opções: {formatted_options}"
        
        # Resetar o contador de tentativas se a resposta for válida
        context['fallback_count'] = 0
        
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
                        "response": "Ótimo! Como posso ajudá-lo agora? Posso auxiliar com:\n\n- Redefinição de senhas\n- Problemas com impressora\n- Configuração de e-mail",
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
                # Adicionar reconhecimento das opções do menu
                intent, confidence = self.classify_intent_advanced(message)
                if intent in ['senha', 'impressora', 'email']:
                    flow_mapping = {
                        'senha': 'password_recovery',
                        'impressora': 'printer_troubleshooting',
                        'email': 'email_configuration',
                    }
                    flow_name = flow_mapping.get(intent)
                    response = self.inference_engine.start_interactive_flow(flow_name, user_id)
                    return {
                        "response": response,
                        "type": "interactive_flow_start",
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
        if intent in ['senha', 'impressora', 'email']:
            # Mapear intenções para fluxos
            flow_mapping = {
                'senha': 'password_recovery', 
                'impressora': 'printer_troubleshooting',
                'email': 'email_configuration',
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
                "response": "Olá! Sou seu assistente de suporte técnico com IA avançada. Como posso ajudá-lo hoje?\n\nPosso auxiliar com:\n- Redefinição de senhas\n- Problemas com impressora\n- Configuração de e-mail",
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
                fallback_response = f"Não entendi completamente, mas notei que você mencionou algo sobre {topics_text}. Você está com problemas relacionados a isso?\n\nPosso ajudar com: redefinição de senhas, problemas com impressora ou configuração de e-mail."
            else:
                fallback_response = f"Desculpe, não consegui entender sua pergunta com confiança suficiente (confiança: {confidence:.1f}).\n\nPosso ajudar com:\n- Redefinição de senhas\n- Problemas com impressora\n- Configuração de e-mail\n\nPode reformular sua pergunta ou ser mais específico?"
            
            return {
                "response": fallback_response,
                "type": "unknown",
                "confidence": confidence
            }
        
        # python src/main.py