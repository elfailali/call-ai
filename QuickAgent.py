import asyncio
import subprocess
import time
import re
import ssl
from openai import OpenAI

context = ssl.create_default_context()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)


class LanguageModelProcessor:
    def __init__(self):
        # self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=self.groq_api_key)
        # self.llm = ChatGroq(temperature=1.2, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768", max_tokens="128")
        self.llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k", max_tokens=256,
                              openai_api_key=OPENAI_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt_v2.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("user query: {text} \n assistant answer:")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def clean_response(self, text):
        cleaned_text = re.sub(r'(<s>|</s>|`|"\[.*?\]"|"\[.*?\]")', '', text)
        return cleaned_text.strip()

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start_time = time.time()
        response = self.conversation.invoke({"text": text})
        execution_time = time.time() - start_time
        cleaned_response = self.clean_response(response['text'])
        self.memory.chat_memory.add_ai_message(cleaned_response)
        print("--> GENERATING TOOK: ", execution_time)
        return cleaned_response


class TextToSpeechOpenAI:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def speak(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text,
        )

        # Guardar el audio como un archivo temporal
        with open("temp_audio.mp3", "wb") as audio_file:
            audio_file.write(response.content)

        print("-> AUDIO FILE SAVED WITH SUCCESS")

        # Reproducir el audio utilizando ffplay.exe
        player_command = ["ffplay.exe", "-nodisp", "-autoexit", "temp_audio.mp3"]
        subprocess.run(player_command)


class TranscriptCollector:
    def __init__(self):
        self.transcript_parts = []

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)


transcript_collector = TranscriptCollector()


async def get_transcript(callback):
    transcription_complete = asyncio.Event()
    config = DeepgramClientOptions(options={"keepalive": "true"})
    deepgram = DeepgramClient("42da42105cb7ca70713a56ab7e846f4868af5653", config)
    dg_connection = deepgram.listen.asynclive.v("1")
    print("Listening...")

    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if not result.speech_final:
            transcript_collector.add_part(sentence)
        else:
            transcript_collector.add_part(sentence)
            full_sentence = transcript_collector.get_full_transcript().strip()
            if full_sentence:
                print(f"Human: {full_sentence}")
                callback(full_sentence)
                transcript_collector.reset()
                transcription_complete.set()

    def on_metadata(self, metadata, **kwargs):
        print(f"\n\n{metadata}\n\n")

    def on_error(self, error, **kwargs):
        print(f"\n\n{error}\n\n")

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model="nova-2",
        punctuate=True,
        language="es-419",
        # language="en-US",
        encoding="linear16",
        channels=1,
        sample_rate=16000,
        endpointing='300',
        smart_format=True,
    )

    await dg_connection.start(options)
    microphone = Microphone(dg_connection.send)
    microphone.start()
    await transcription_complete.wait()
    microphone.finish()
    await dg_connection.finish()


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeechOpenAI()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            await get_transcript(handle_full_sentence)
            if "bye." in self.transcription_response.lower():
                end_response = "Goodbye, have a great day."
                self.tts.speak(end_response)
                break
            llm_response = self.llm.process(self.transcription_response)
            print("LLM RESPONSE:\n ", llm_response)
            self.tts.speak(llm_response)
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
