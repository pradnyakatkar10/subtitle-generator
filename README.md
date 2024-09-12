# subtitle-generator #
A) Details of Team members
I) Sejal Yadav - ysejal7352@gmail.com (ECS 2nd Year)
II) Nidhi Tripathi - tripathinidhi172@gmail.com (ECS 2nd Year)
III) Devanshi Singh - devanshisingh228@gmail. (ECS 2nd Year)
IV) Pradnya Katkar - katkarpradnya184@gmail.com

B) Introduction
The project, titled “Subtitle generation with language translation," is designed to facilitate the creation of accessible and multilingual video content. By automating the process of adding precise subtitles and translations to video materials, this project empowers content creators to broaden their viewership, enhance user experience, and provide inclusive content to a global audience. This project combines speech recognition and video editing techniques to streamline a traditionally labor-intensive process. The initial focus is on commonly spoken languages, with the potential for future expansion.

C)Dataset Description

For this project, no external dataset was used. The system processes video files directly, extracting audio from the video and converting it into text using a speech-to-text model (Whisper). The video files can be in various formats, such as MP4, AVI, or MKV. The extracted audio is processed to generate accurate subtitles, which can then be translated into different languages. This allows the project to work dynamically with any video file provided as input by the user.

D) Dataset Split Info

Since this project does not involve a dataset that requires splitting (such as training and testing data), it handles video inputs directly. The process involves the following steps:
--A video file is uploaded or provided as input.
--The system extracts the audio track from the video file using ffmpeg.
--The audio is then transcribed into text using the Whisper model for speech-to-text conversion.
--Subtitles are generated and, if necessary, translated into the selected language.
Since the input is real-time video/audio processing, there is no need for splitting the data into training and testing sets.

E) Approach
The project follows a structured approach to generate subtitles from video files and, if needed, translate them into other languages. Here’s how the approach works step-by-step:

--Input Video Handling:
The user uploads a video file in a supported format (e.g., MP4, AVI).
ffmpeg is used to extract the audio track from the video. This step prepares the audio for transcription.

--Speech-to-Text Conversion (Whisper Model):
The extracted audio is processed using Whisper, a powerful speech recognition model.
Whisper transcribes the speech in the audio into text, producing raw subtitles with accurate timestamps for each spoken segment.

--Language Detection and Translation (Optional):
Whisper automatically detects the spoken language in the video.
If the user requests a translation, the project uses a LLaMA-based translation model to convert the transcribed text into the target language.
The translation process ensures that subtitles are not only accurate but also accessible to users in multiple languages.

--SRT File Generation:
Once the transcription (and translation, if applicable) is complete, the system generates a subtitle file in SRT format.
The SRT file contains the transcribed or translated text along with timestamps to ensure accurate synchronization with the video.

--Video Output:
If the user chooses, the system can overlay the subtitles onto the video using ffmpeg to produce a subtitled video file.
The subtitled video is saved in the specified output directory with the corresponding language details.

--GUI for User Interaction (Gooey):
A graphical interface built with Gooey allows users to interact with the project easily.
Users can select video files, choose the language for transcription and translation, and configure output options without needing to run commands manually.

This approach allows for flexible, dynamic subtitle generation and translation, making video content more accessible across different languages.


F) Results

User Interface: 
![image](https://github.com/user-attachments/assets/c09c8652-7e3e-4843-bdca-b983333f4238)
![image](https://github.com/user-attachments/assets/fae26aa3-074b-404a-ac54-3f71a8c284fb)
![image](https://github.com/user-attachments/assets/1ebaa3f7-e528-4f7c-aa2d-5dcf20ec8229)
![image](https://github.com/user-attachments/assets/b9b58cb9-2b60-4766-9622-6e90c9d7adeb)

SRT file for user:
![image](https://github.com/user-attachments/assets/9cb5d664-e724-49e2-8047-b80e9b6858cf)
![image](https://github.com/user-attachments/assets/5ba774a8-671e-4a47-9ee9-bccfe718aa8c)

The project successfully generates accurate subtitles for video files. The results can be broken down into the following key outcomes:

Transcription Accuracy:
The Whisper model effectively transcribes speech from video files, generating text that aligns closely with the spoken content. This ensures that subtitles are clear and understandable.
Timestamps are automatically added, synchronizing the subtitles with the audio in the video.

Translation Success (Optional):
When a translation is requested, the project provides translated subtitles that match the original content in the target language, enabling accessibility for multilingual audiences.

SRT File Generation:
The project outputs the transcription or translation in SRT format, a widely supported subtitle format. This file can be used in various video players, editors, or as a standalone file for content distribution.

Subtitled Video Output (Optional):
If the user chooses to overlay subtitles on the video, the project generates a new video file with the subtitles burned into the video. This file is ready for distribution, making it ideal for sharing across platforms.

G) Dependencies
--ffmpeg-python
--torch
--transformers
--moviepy
--googletrans
--whisper

H) Performance and Accuracy

--Transcription Accuracy: Whisper model has a high accuracy in transcribing clear speech, achieving over 90% accuracy in standard environments.
--Translation Accuracy: Llama2 provides reliable translations, but some nuances in certain languages may be lost. Further tuning or custom translation models might improve this.
--Speed: On a typical system (8GB RAM, Intel i5), the subtitle generation for a 5-minute video takes about 2-3 minutes.
--Efficiency: Processing speed and accuracy may decrease with poor audio quality, heavy accents, or multiple overlapping speakers.
--Limitations: Translations may not be 100% contextually accurate. Heavy background noise can reduce transcription accuracy.

I) F1 Score
The F1 Score is typically used to evaluate classification models. Since this project focuses on transcription and translation, the F1 Score is not applicable. Instead, accuracy and real-time performance are better measures for evaluating the success of the project.

J) Novelty/Factor
--Integration of AI for Real-Time Transcription and Translation: This project uses Whisper AI and Llama2 for real-time speech-to-text and multi-language translation.
--User-Friendly GUI: A Gooey-based GUI is implemented to make it easier for users to upload videos, select languages, and view generated subtitles without needing to interact with the command line.
--Real-Time Subtitle Generation: The project aims to generate subtitles in real-time, allowing users to quickly process videos without significant delays.
--Multi-Language Support: Automatic translation into multiple languages is integrated, expanding accessibility to global users.

K) 


L) Refrences:

1] Gooey GitHub (https://github.com/chriskiehl/Gooey)
2]	https://www.irjet.net/archives/V7/i5/IRJET-V7I51463.pdfTutorialspoint: - 
3]	https://hv.diva-portal.org/smash/get/diva2:241802/FULLTEXT01.pdf 
4]	https://international-dubbing.fandom.com/wiki/International_Dubbing_Wiki 
5]	https://aclanthology.org/P16-1029.pdf 
6]	https://www.opensubtitles.org/en/search/subs 
7]	https://ieeexplore.ieee.org/document/9773697 
8]  Whisper AI GitHub (https://github.com/openai/whisper)
9] ffmpeg Documentation(https://ffmpeg.org/documentation.html)
10] Googletrans Documentation (https://py-googletrans.readthedocs.io/en/latest/)
11] AssemblyAI API (https://www.assemblyai.com/docs/)

