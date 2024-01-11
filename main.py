from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import openai
import os
from dotenv.main import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

os.environ["OPENAI_API_KEY"] = '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'
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text = data.get('data')
    user_input = text
    try:
        conversation = ConversationChain(llm=llm, memory=memory)
        output = conversation.predict(input=user_input)
        memory.save_context({"input": user_input}, {"output": output})
        return jsonify({"response": True, "message": output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
