#!/usr/bin/env python3
"""
주식 분석 AI 에이전트 테스트 및 실행 관련 함수들
"""

import os
import traceback
from agent import create_stock_agent
from prompts import ERROR_MESSAGES

def test_simple_agent():
    """간단한 6단계 에이전트 테스트"""
    
    # 환경 변수 확인
    if not os.getenv("CLOVASTUDIO_API_KEY"):
        print(ERROR_MESSAGES["api_key_missing"])
        return
    
    print("🧪 6단계 개선된 계획 수립 에이전트 테스트 시작...")
    
    graph = create_stock_agent()
    
    # 간단한 테스트 케이스
    test_question = "오늘 날짜를 알려주세요"
    
    print(f"\n🧪 테스트: {test_question}")
    print("-" * 50)
    
    try:
        # 초기 상태 설정
        initial_state = {
            "user_input": test_question,
            "intention_analysis": "",
            "text_plan": "",
            "task_list": [],  # 추가
            "step_by_step_plan": [],  # 단계별 계획 추출 결과
            "plan_structure": {},  # 계획 구조 정보
            "detailed_plan": [],
            "plan_validation_result": "",
            "plan_validation_status": "",
            "tool_validation_result": "",
            "tool_validation_status": "",
            "revised_plan": "",
            "revised_tools": "",
            "revision_count": 0,
            "task_execution_results": [],
            "execution_summary": {},
            "final_answer": "",
            "error_message": "",
            "current_stage": ""
        }
        
        config = {"configurable": {"thread_id": "test_simple"}}
        result = graph.invoke(initial_state, config)
        
        print(f"\n✅ 최종 응답: {result.get('final_answer', '응답을 가져올 수 없습니다.')}")
        
        print("\n📊 6단계 프로세스 확인:")
        print(f"✓ 1단계: 의도 파악 - {result.get('intention_analysis', '')[:100]}...")
        
        # text_plan 확인
        text_plan = result.get('text_plan', '')
        if text_plan:
            text_plan_summary = text_plan[:100].replace('\n', ' ')
            print(f"✓ 2단계: 텍스트 계획 - {text_plan_summary}...")
        else:
            print("✓ 2단계: 텍스트 계획 - 계획 없음...")
        
        # 단계별 계획 추출 결과 확인
        step_by_step_plan = result.get('step_by_step_plan', [])
        plan_structure = result.get('plan_structure', {})
        if step_by_step_plan:
            total_steps = plan_structure.get('total_steps', len(step_by_step_plan))
            complexity = plan_structure.get('complexity_level', 'low')
            print(f"✓ 2b단계: 단계별 계획 추출 - {total_steps}개 단계 (복잡도: {complexity})...")
        else:
            print("✓ 2b단계: 단계별 계획 추출 - 단계 없음...")
        
        # detailed_plan 확인
        detailed_plan = result.get('detailed_plan', [])
        if detailed_plan:
            plan_summary = ", ".join([f"{task.get('tool_name', 'no-tool')}-{task.get('task_description', 'no-desc')}" for task in detailed_plan])[:100]
            print(f"✓ 3단계: 계획 구체화 - {plan_summary}...")
        else:
            print("✓ 3단계: 계획 구체화 - 계획 없음...")
        
        # 검증 결과 확인
        plan_validation = result.get('plan_validation_result', '')
        tool_validation = result.get('tool_validation_result', '')
        if plan_validation or tool_validation:
            validation_summary = f"계획검증: {plan_validation[:30] if plan_validation else '없음'}, 도구검증: {tool_validation[:30] if tool_validation else '없음'}"
            print(f"✓ 4단계: 계획 검증 - {validation_summary}...")
        else:
            print("✓ 4단계: 계획 검증 - 검증 없음...")
        
        # 실행 요약 통계 표시
        execution_summary = result.get('execution_summary', {})
        if execution_summary:
            total_tasks = execution_summary.get('total_tasks', 0)
            successful_tasks = execution_summary.get('successful_tasks', 0)
            failed_tasks = execution_summary.get('failed_tasks', 0)
            total_duration = execution_summary.get('total_duration_seconds', 0)
            revision_count = result.get('revision_count', 0)
            print(f"✓ 5단계: 작업 수행 - {total_tasks}개 작업 (성공: {successful_tasks}, 실패: {failed_tasks}, 소요시간: {total_duration}초)")
            print(f"✓ 6단계: 결과 출력 - 완료 (수정횟수: {revision_count}회)")
        else:
            print(f"✓ 5단계: 작업 수행 - {len(result.get('task_execution_results', []))}개 작업 실행")
            print(f"✓ 6단계: 결과 출력 - 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


def test_advanced_questions():
    """고급 질문으로 6단계 에이전트 테스트"""
    
    # 환경 변수 확인
    if not os.getenv("CLOVASTUDIO_API_KEY"):
        print(ERROR_MESSAGES["api_key_missing"])
        return
    
    print("🚀 6단계 고급 질문 테스트 시작...")
    
    graph = create_stock_agent()
    
    advanced_questions = [
        "RSI 30 이하인 KOSPI 종목을 찾아주세요",
        "골든크로스가 발생한 종목을 알려주세요",
        "최근 거래량이 급증한 종목을 분석해주세요"
    ]
    
    for i, question in enumerate(advanced_questions, 1):
        print(f"\n🧪 고급 테스트 {i}: {question}")
        print("-" * 60)
        
        try:
            initial_state = {
                "user_input": question,
                "intention_analysis": "",
                "text_plan": "",
                "task_list": [],  # 추가
                "step_by_step_plan": [],  # 단계별 계획 추출 결과
                "plan_structure": {},  # 계획 구조 정보
                "detailed_plan": [],
                "plan_validation_result": "",
                "plan_validation_status": "",
                "tool_validation_result": "",
                "tool_validation_status": "",
                "revised_plan": "",
                "revised_tools": "",
                "revision_count": 0,
                "task_execution_results": [],
                "execution_summary": {},
                "final_answer": "",
                "error_message": "",
                "current_stage": ""
            }
            
            config = {"configurable": {"thread_id": f"test_advanced_{i}"}}
            result = graph.invoke(initial_state, config)
            
            print(f"\n✅ 응답: {result.get('final_answer', '응답을 생성할 수 없습니다.')}")
            
            # 계획 수정 횟수 표시
            revision_count = result.get('revision_count', 0)
            if revision_count > 0:
                print(f"📝 계획 수정 횟수: {revision_count}회")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {str(e)}")
        
        print("\n" + "="*60)


def interactive_chat():
    """대화형 챗봇"""
    print("🤖 주식 분석 AI 에이전트에 오신 것을 환영합니다! (6단계 개선된 버전)")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.\n")
    
    # 환경 변수 확인
    if not os.getenv("CLOVASTUDIO_API_KEY"):
        print(ERROR_MESSAGES["api_key_missing"])
        return
    
    graph = create_stock_agent()
    
    while True:
        try:
            user_input = input("\n💭 질문을 입력하세요: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 안녕히 가세요!")
                break
            
            if not user_input:
                continue
            
            print("\n🤔 6단계 분석 과정 시작...")
            
            initial_state = {
                "user_input": user_input,
                "intention_analysis": "",
                "text_plan": "",
                "task_list": [],  # 추가
                "step_by_step_plan": [],  # 단계별 계획 추출 결과
                "plan_structure": {},  # 계획 구조 정보
                "detailed_plan": [],
                "plan_validation_result": "",
                "plan_validation_status": "",
                "tool_validation_result": "",
                "tool_validation_status": "",
                "revised_plan": "",
                "revised_tools": "",
                "revision_count": 0,
                "task_execution_results": [],
                "execution_summary": {},
                "final_answer": "",
                "error_message": "",
                "current_stage": ""
            }
            
            config = {"configurable": {"thread_id": "interactive_chat"}}
            result = graph.invoke(initial_state, config)
            
            print(f"\n🤖 답변: {result.get('final_answer', '응답을 생성할 수 없습니다.')}")
            
            # 추가 정보 표시
            revision_count = result.get('revision_count', 0)
            if revision_count > 0:
                print(f"\n📊 프로세스 정보: 계획 수정 {revision_count}회")
            
        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"❌ 오류 발생: {str(e)}")


def main():
    """메인 실행 함수"""
    # 환경 변수 로드
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv가 설치되지 않았습니다.")
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "chat":
            interactive_chat()
        elif sys.argv[1] == "advanced":
            test_advanced_questions()
        else:
            test_simple_agent()
    else:
        test_simple_agent()


if __name__ == "__main__":
    main() 