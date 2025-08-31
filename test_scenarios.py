#!/usr/bin/env python3
"""
Script de testes para o chatbot de suporte técnico com IA.
Avalia correção, desempenho e limitações do sistema.
"""

import requests
import json
import time
from typing import List, Dict, Any

class ChatbotTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Testa funcionalidades básicas do chatbot"""
        print("=== Testando Funcionalidades Básicas ===")
        
        test_cases = [
            {
                "name": "Saudação",
                "input": "Olá",
                "expected_type": "greeting",
                "expected_keywords": ["assistente", "ajudar"]
            },
            {
                "name": "Configuração Wi-Fi",
                "input": "Como configurar Wi-Fi?",
                "expected_type": "solution",
                "expected_keywords": ["configurações", "rede", "senha"]
            },
            {
                "name": "Redefinição de Senha",
                "input": "Esqueci minha senha",
                "expected_type": "solution",
                "expected_keywords": ["redefinir", "email", "link"]
            },
            {
                "name": "Problema com Impressora",
                "input": "Impressora não funciona",
                "expected_type": "solution",
                "expected_keywords": ["impressora", "ligada", "drivers"]
            },
            {
                "name": "Configuração de Email",
                "input": "Como configurar email?",
                "expected_type": "solution",
                "expected_keywords": ["email", "conta", "servidor"]
            },
            {
                "name": "Despedida",
                "input": "Obrigado, tchau",
                "expected_type": "farewell",
                "expected_keywords": ["obrigado", "dia"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                response = self._send_message(test_case["input"])
                
                # Verificar tipo de resposta
                type_correct = response.get("response_type") == test_case["expected_type"]
                
                # Verificar palavras-chave na resposta
                bot_response = response.get("bot_response", "").lower()
                keywords_found = sum(1 for keyword in test_case["expected_keywords"] 
                                   if keyword in bot_response)
                keywords_score = keywords_found / len(test_case["expected_keywords"])
                
                result = {
                    "test_name": test_case["name"],
                    "input": test_case["input"],
                    "response": bot_response,
                    "type_correct": type_correct,
                    "keywords_score": keywords_score,
                    "confidence": response.get("confidence", 0),
                    "passed": type_correct and keywords_score >= 0.5
                }
                
                results.append(result)
                print(f"✓ {test_case['name']}: {'PASSOU' if result['passed'] else 'FALHOU'}")
                
            except Exception as e:
                print(f"✗ {test_case['name']}: ERRO - {str(e)}")
                results.append({
                    "test_name": test_case["name"],
                    "error": str(e),
                    "passed": False
                })
        
        return {
            "test_type": "basic_functionality",
            "results": results,
            "pass_rate": sum(1 for r in results if r.get("passed", False)) / len(results)
        }
    
    def test_diagnostic_system(self) -> Dict[str, Any]:
        """Testa o sistema de diagnóstico interativo"""
        print("\n=== Testando Sistema de Diagnóstico ===")
        
        # Teste 1: Iniciar diagnóstico Wi-Fi
        try:
            response1 = self._send_message("diagnóstico wifi")
            diagnostic_started = "perguntas" in response1.get("bot_response", "").lower()
            
            if diagnostic_started:
                # Simular respostas ao diagnóstico
                responses = ["sim", "não", "sim", "sim"]
                final_response = None
                
                for i, answer in enumerate(responses):
                    response = self._send_message(answer)
                    if i == len(responses) - 1:  # Última resposta
                        final_response = response
                
                solution_provided = final_response and "recomendo" in final_response.get("bot_response", "").lower()
                
                result = {
                    "test_name": "Diagnóstico Wi-Fi Completo",
                    "diagnostic_started": diagnostic_started,
                    "solution_provided": solution_provided,
                    "final_response": final_response.get("bot_response", "") if final_response else "",
                    "passed": diagnostic_started and solution_provided
                }
            else:
                result = {
                    "test_name": "Diagnóstico Wi-Fi Completo",
                    "diagnostic_started": False,
                    "passed": False
                }
            
            print(f"✓ Diagnóstico Wi-Fi: {'PASSOU' if result['passed'] else 'FALHOU'}")
            
        except Exception as e:
            print(f"✗ Diagnóstico Wi-Fi: ERRO - {str(e)}")
            result = {
                "test_name": "Diagnóstico Wi-Fi Completo",
                "error": str(e),
                "passed": False
            }
        
        return {
            "test_type": "diagnostic_system",
            "results": [result],
            "pass_rate": 1.0 if result.get("passed", False) else 0.0
        }
    
    def test_nlp_capabilities(self) -> Dict[str, Any]:
        """Testa capacidades de PLN (variações de linguagem)"""
        print("\n=== Testando Capacidades de PLN ===")
        
        nlp_test_cases = [
            {
                "name": "Variações Wi-Fi",
                "inputs": [
                    "Como configurar wifi?",
                    "Configuração de wi-fi",
                    "Ajuda com internet",
                    "Conectar na rede wireless"
                ],
                "expected_topic": "wifi"
            },
            {
                "name": "Variações Senha",
                "inputs": [
                    "Perdi minha senha",
                    "Não lembro da password",
                    "Resetar senha",
                    "Como recuperar acesso?"
                ],
                "expected_topic": "senha"
            },
            {
                "name": "Linguagem Informal",
                "inputs": [
                    "Oi, tudo bem?",
                    "E aí!",
                    "Olá, como vai?",
                    "Bom dia!"
                ],
                "expected_type": "greeting"
            }
        ]
        
        results = []
        for test_group in nlp_test_cases:
            group_results = []
            
            for input_text in test_group["inputs"]:
                try:
                    response = self._send_message(input_text)
                    
                    if "expected_topic" in test_group:
                        correct = response.get("topic") == test_group["expected_topic"]
                    else:
                        correct = response.get("response_type") == test_group["expected_type"]
                    
                    group_results.append({
                        "input": input_text,
                        "response_type": response.get("response_type"),
                        "topic": response.get("topic"),
                        "confidence": response.get("confidence", 0),
                        "correct": correct
                    })
                    
                except Exception as e:
                    group_results.append({
                        "input": input_text,
                        "error": str(e),
                        "correct": False
                    })
            
            group_pass_rate = sum(1 for r in group_results if r.get("correct", False)) / len(group_results)
            
            results.append({
                "test_name": test_group["name"],
                "individual_results": group_results,
                "pass_rate": group_pass_rate,
                "passed": group_pass_rate >= 0.7  # 70% de acerto
            })
            
            print(f"✓ {test_group['name']}: {group_pass_rate:.1%} de acerto")
        
        return {
            "test_type": "nlp_capabilities",
            "results": results,
            "pass_rate": sum(1 for r in results if r.get("passed", False)) / len(results)
        }
    
    def test_performance(self) -> Dict[str, Any]:
        """Testa performance do sistema"""
        print("\n=== Testando Performance ===")
        
        test_messages = [
            "Olá",
            "Como configurar Wi-Fi?",
            "Esqueci minha senha",
            "Problema com impressora",
            "diagnóstico wifi"
        ]
        
        response_times = []
        
        for message in test_messages:
            try:
                start_time = time.time()
                response = self._send_message(message)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
            except Exception as e:
                print(f"Erro ao testar performance com '{message}': {e}")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # Critério: resposta em menos de 2 segundos em média
            performance_good = avg_response_time < 2.0
            
            print(f"✓ Tempo médio de resposta: {avg_response_time:.2f}s")
            print(f"✓ Tempo máximo: {max_response_time:.2f}s")
            print(f"✓ Tempo mínimo: {min_response_time:.2f}s")
            
            return {
                "test_type": "performance",
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "min_response_time": min_response_time,
                "performance_good": performance_good,
                "passed": performance_good
            }
        else:
            return {
                "test_type": "performance",
                "error": "Não foi possível medir performance",
                "passed": False
            }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Testa casos extremos e limitações"""
        print("\n=== Testando Casos Extremos ===")
        
        edge_cases = [
            {
                "name": "Mensagem Vazia",
                "input": "",
                "should_handle_gracefully": True
            },
            {
                "name": "Mensagem Muito Longa",
                "input": "a" * 1000,
                "should_handle_gracefully": True
            },
            {
                "name": "Caracteres Especiais",
                "input": "!@#$%^&*()",
                "should_handle_gracefully": True
            },
            {
                "name": "Pergunta Ambígua",
                "input": "Não funciona",
                "should_handle_gracefully": True
            },
            {
                "name": "Idioma Estrangeiro",
                "input": "Hello, how are you?",
                "should_handle_gracefully": True
            }
        ]
        
        results = []
        for case in edge_cases:
            try:
                response = self._send_message(case["input"])
                
                # Verificar se o sistema respondeu sem erro
                handled_gracefully = "error" not in response and response.get("bot_response")
                
                result = {
                    "test_name": case["name"],
                    "input": case["input"],
                    "handled_gracefully": handled_gracefully,
                    "response": response.get("bot_response", ""),
                    "passed": handled_gracefully
                }
                
                results.append(result)
                print(f"✓ {case['name']}: {'PASSOU' if result['passed'] else 'FALHOU'}")
                
            except Exception as e:
                print(f"✗ {case['name']}: ERRO - {str(e)}")
                results.append({
                    "test_name": case["name"],
                    "error": str(e),
                    "passed": False
                })
        
        return {
            "test_type": "edge_cases",
            "results": results,
            "pass_rate": sum(1 for r in results if r.get("passed", False)) / len(results)
        }
    
    def _send_message(self, message: str) -> Dict[str, Any]:
        """Envia mensagem para o chatbot e retorna resposta"""
        url = f"{self.base_url}/api/chat"
        payload = {"message": message}
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Executa todos os testes"""
        print("🤖 Iniciando Testes do Chatbot de Suporte Técnico com IA")
        print("=" * 60)
        
        # Verificar se o servidor está rodando
        try:
            health_response = requests.get(f"{self.base_url}/api/health", timeout=5)
            health_response.raise_for_status()
            print("✓ Servidor está online e funcionando")
        except Exception as e:
            print(f"✗ Erro ao conectar com o servidor: {e}")
            return {"error": "Servidor não está acessível"}
        
        # Executar todos os testes
        test_results = {
            "basic_functionality": self.test_basic_functionality(),
            "diagnostic_system": self.test_diagnostic_system(),
            "nlp_capabilities": self.test_nlp_capabilities(),
            "performance": self.test_performance(),
            "edge_cases": self.test_edge_cases()
        }
        
        # Calcular estatísticas gerais
        total_tests = 0
        passed_tests = 0
        
        for test_type, results in test_results.items():
            if "pass_rate" in results:
                if isinstance(results["results"], list):
                    total_tests += len(results["results"])
                    passed_tests += int(results["pass_rate"] * len(results["results"]))
                else:
                    total_tests += 1
                    passed_tests += 1 if results.get("passed", False) else 0
        
        overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES")
        print("=" * 60)
        print(f"Total de testes: {total_tests}")
        print(f"Testes aprovados: {passed_tests}")
        print(f"Taxa de aprovação: {overall_pass_rate:.1%}")
        
        if overall_pass_rate >= 0.8:
            print("🎉 RESULTADO: EXCELENTE - Sistema funcionando muito bem!")
        elif overall_pass_rate >= 0.6:
            print("✅ RESULTADO: BOM - Sistema funcionando adequadamente")
        elif overall_pass_rate >= 0.4:
            print("⚠️  RESULTADO: REGULAR - Sistema precisa de melhorias")
        else:
            print("❌ RESULTADO: RUIM - Sistema precisa de correções significativas")
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_pass_rate": overall_pass_rate,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return test_results

def main():
    """Função principal para executar os testes"""
    tester = ChatbotTester()
    results = tester.run_all_tests()
    
    # Salvar resultados em arquivo
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Resultados salvos em: test_results.json")

if __name__ == "__main__":
    main()

