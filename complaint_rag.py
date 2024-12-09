import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



import os
from pathlib import Path
from dotenv import load_dotenv

# 현재 작업 디렉토리 확인
print("현재 작업 디렉토리:", os.getcwd())

# 현재 디렉토리의 .env 파일 경로 지정
env_path = Path('.') / '.env'
load_dotenv(override=True)  # override=True를 사용하여 기존 값을 덮어씁니다

# 환경변수 확인
api_key = os.getenv('OPENAI_API_KEY')
print("OPENAI_API_KEY:", api_key)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

import os
import json

# 디렉터리 내 모든 JSON 파일 불러오기
def load_all_json_files(directory_path):
    all_cases = []  # 모든 판례 데이터를 저장할 리스트

    # 디렉터리의 모든 하위 폴더 및 파일 탐색
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):  # JSON 파일만 선택
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        all_cases.extend(data)  # 파일의 데이터를 리스트에 추가
                    except json.JSONDecodeError as e:
                        print(f"JSON 디코딩 실패: {file_path}, 오류: {e}")

    return all_cases

# JSON 파일 경로 (최상위 폴더)
directory_path = "C:/Users/SeoyeonKim/Documents/hanyangUni/4th_grade/1st_sem/AI_application/dev/RAG_dataset"

# 모든 JSON 파일의 데이터 로드
cases = load_all_json_files(directory_path)

# 데이터 확인
print(f"불러온 판례 수: {len(cases)}")
for case in cases[:5]:  # 일부 데이터를 출력
    print(case)

# 벡터화된 판례 저장 (FAISS에 저장)
texts = [case["text"] for case in cases]
metadata = [{"category": case["category"], "settlement": case["settlement"], "sentence": case["sentence"]} for case in cases]

vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadata)

# RAG 시스템 구축
def build_rag_system(vector_store):
    """RAG 시스템을 구축하는 함수"""

    # 검색기(Retriever) 설정
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 검색할 문서 수

    # 프롬프트 템플릿 설정
    template = """당신은 고소 사건을 전문으로 한 변호사입니다.
    의뢰인을 자신이 겪은 사건으로 고소장을 작성하고 싶어합니다.
    주어진 문서 내용을 바탕으로 질문에 답변해주세요.
    문서에 없는 내용이면 "판례를 찾을 수 없습니다."라고 답변해주세요.
    
    문서 내용:
    {context}
    
    질문:
    {question}
    
    답변:"""
    prompt = PromptTemplate.from_template(template)

    # LLM 설정 (gpt-4 모델을 사용)
    llm = ChatOpenAI(
        temperature=0,  # 응답의 창의성 수준 (0: 일관된 응답)
        model_name="gpt-4"  # 사용할 모델
    )

    # RAG 체인 구성
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",  # 텍스트를 모두 읽고 응답 생성
        retriever=retriever,
        return_source_documents=True  # 반환되는 문서도 포함
    )
    
    return chain

# chain을 생성하여 사용
chain = build_rag_system(vector_store)

#category = "성추행"

# 사용자 질문
query = """

중고거래로 모니터를 구매했는데 모니터가 아니라 벽돌이 왔어요. 어떻게 해야할까요?

"""

# 질의 실행
result = chain({"query": query})

# 출력
print("생성된 답변:")
print(result["result"])

print("\n참조된 판례:")
for doc in result["source_documents"]:
    print(f"- 카테고리: {doc.metadata['category']}")
    print(f"  내용: {doc.page_content}")
    print(f"  합의금: {doc.metadata['settlement']}원")
    print(f"  형량: {doc.metadata['sentence']}개월")

