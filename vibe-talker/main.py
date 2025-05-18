import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from app_graph import create_chat_graph
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import asyncio

MONGODB_URI = "mongodb://localhost:27017"
config = {
    "configurable": {
        "thread_id": "101"
    }
}

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        # Create a chat graph with the checkpointer
        graph = create_chat_graph(checkpointer)

        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 1

            while True:
                print("Please say something:")
                audio = r.listen(source)

                try:
                    print("Processing audio...")

                    sst = r.recognize_google(audio)
                    print("You said: ", sst)

                    messages = [{"role": "user", "content": sst}]
                    events = graph.stream({ "messages": messages }, config, stream_mode="values")

                    for event in events:
                        if "messages" in event:
                            event["messages"][-1].pretty_print()

                except sr.UnknownValueError:
                    print("Sorry, I did not understand that.")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")


openai = AsyncOpenAI()

async def speak(text: str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    # asyncio.run(speak("This is the sample voice hi Om"))
    main()