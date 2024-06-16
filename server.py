import torch

from segments import instruction_chunks, question_embeddings, unique_definitions
from search_models import get_embedding, re_rank
from saiga import Saiga
saiga = Saiga()

from flask import Flask, request
from copy import deepcopy

def get_useful_instructions(search_query):
    query_embeddings = get_embedding(search_query)
    useful_instructions = deepcopy(sorted(instruction_chunks, key=lambda x: -(x['embedding'] @ query_embeddings.T).item())[:20])
    return useful_instructions


def get_similar_questions(search_query):
    query_embeddings = get_embedding(search_query)
    useful_instructions = deepcopy(sorted(question_embeddings, key=lambda x: -(x['embedding'] @ query_embeddings.T).item())[:20])
    return useful_instructions


def get_search_query(messages):
    search_query = saiga.chat([
        {'role': 'system',
         'content': 'Преобразовывай запрос клиента в запрос к поисковой системе с учетом истории диалога, чтобы лучше находить релевантные документы. Пиши только перефразированный запрос и ничего лишнего.'},
        *messages
    ])

    return search_query


def preprocess_messages(messages):
    for mes in messages:
        if mes['role'] == 'user':
            for term, definition in unique_definitions.items():
                lower_term = term.lower()
                mes['content'] = mes['content'].replace(term, f'{term} ({definition})')
                mes['content'] = mes['content'].replace(lower_term, f'{lower_term} ({definition})')

    return messages


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    messages = preprocess_messages(request.json['messages'][-7:])

    with torch.no_grad():
        search_query = get_search_query(messages)

        useful_instructions = get_useful_instructions(search_query)
        useful_instructions = re_rank(
            search_query, [item['text'] for item in useful_instructions], useful_instructions, return_num=3
        )

        for inst in useful_instructions:
            del inst['embedding']

            inst['modified_text'] = saiga.chat([
                {'role': 'system',
                 'content': f'Выдели из запроса пользователя фрагмент текста, где содержится ответ на вопрос: {search_query}. Не пиши ничего лишнего, только выделенный фрагмент. Если такого фрагмента нет, напиши "Нет фрагмента".'},
                {'role': 'user', 'content': inst['text']}
            ])

        useful_instructions = [inst for inst in useful_instructions if 'нет фрагмента' not in inst['modified_text'].lower()]

        instructions_info = '\n'.join([f'Название файла: {inst["filename"]}. Название параграфа: {inst["title"]}. Информация: {inst["modified_text"]}' for inst in useful_instructions])

        if len(set([item['filename'] for item in useful_instructions])) == 1:
            answer = saiga.chat([
                {'role': 'system', 'content': 'По найденной информации из инструкций Росатома ответь на вопрос пользователя, учитывая контекст диалога. Если по найденной информации невозможно однозначно ответить на вопрос пользователя, напиши "Нет ответа". Если ты не уверен в ответе, предложи обратиться к оператору службы поддержки.'},
                {'role': 'assistant', 'content': f'Я нашел следующую информацию: {useful_instructions}'},
                *messages
            ])

        elif instructions_info:
            answer = saiga.chat([
                {'role': 'system', 'content': 'По найденной информации из инструкций Росатома с учетом контекста диалога задай уточняющий вопрос на вопрос пользователя, чтобы можно было однозначно определить, информацию из какого файла с каким названием надо использовать для ответа на вопрос. Если ты не уверен в ответе, предложи обратиться к оператору службы поддержки.'},
                {'role': 'assistant', 'content': f'Я нашел следующую информацию: {useful_instructions}. Готов задать уточняющий вопрос на Ваш запрос или перевести на службу поддержки.'},
                *messages
            ])
        else:
            similar_questions = get_similar_questions(search_query)
            similar_questions = re_rank(
                search_query, [item['question'] for item in similar_questions], similar_questions, return_num=5
            )

            for question in similar_questions:
                del question['embedding']

                question['question_validity'] = saiga.chat([
                    {'role': 'user',
                     'content': f'Определи, является ли Запрос похожим на Текст, пиши только Да или Нет, больше ничего не пиши.\nЗапрос: {search_query}\nТекст: {question["question"]}'}
                ])

            similar_questions = [question for question in similar_questions if 'да' in question['question_validity'].lower()]

            if similar_questions:
                answer = 'Предоставляю ответы на похожие вопросы:\n'

                for question in similar_questions:
                    answer += f'**Вопрос**: {question["question"]}\n**Ответ**:{question["answer"]}\n\n'

                answer += 'Если среди предложенных вариантов нет ответа на Ваш запрос, переформулируйте его или обратитесь к оператору.'

                return {
                    'answer': answer,
                    'useful_instructions': []
                }
            else:
                answer = 'Извините, я не знаю, как ответить на Ваш вопрос. Переформулируйте Ваш вопрос или обратитесь за помощью к оператору.'

        return {
            'answer': answer,
            'useful_instructions': useful_instructions
        }


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
