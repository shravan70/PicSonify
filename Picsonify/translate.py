import gtts


from playsound import playsound

text = input("Enter text to convert to sound: ")
sound = gtts.gTTS(text, lang="en")

print(label)
sound.save("Sound/Welcome.mp3")
playsound("Sound/Welcome.mp3")
