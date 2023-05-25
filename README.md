<div align="center" id="top"> 
  <img src="./.github/app.gif" alt="LangChain Conversation Bot" />

  &#xa0;

  <!-- <a href="https://akshat_workspace.netlify.app">Demo</a> -->
</div>

<h1 align="center">LangChain Conversation Bot</h1>

<p align="center">
  <!-- <img alt="Github top language" src="https://img.shields.io/github/languages/top/{{technickel08}}/akshat_workspace?color=56BEB8"> -->

  <!-- <img alt="Github language count" src="https://img.shields.io/github/languages/count/{{YOUR_GITHUB_USERNAME}}/akshat_workspace?color=56BEB8">

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/{{YOUR_GITHUB_USERNAME}}/akshat_workspace?color=56BEB8">

  <img alt="License" src="https://img.shields.io/github/license/{{YOUR_GITHUB_USERNAME}}/akshat_workspace?color=56BEB8"> -->

  <!-- <img alt="Github issues" src="https://img.shields.io/github/issues/{{YOUR_GITHUB_USERNAME}}/akshat_workspace?color=56BEB8" /> -->

  <!-- <img alt="Github forks" src="https://img.shields.io/github/forks/{{YOUR_GITHUB_USERNAME}}/akshat_workspace?color=56BEB8" /> -->

  <!-- <img alt="Github stars" src="https://img.shields.io/github/stars/{{YOUR_GITHUB_USERNAME}}/akshat_workspace?color=56BEB8" /> -->
</p>

<!-- Status -->

<!-- <h4 align="center"> 
	🚧  Akshat_workspace 🚀 Under construction...  🚧
</h4> 

<hr> -->

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <!-- <a href="#sparkles-features">Features</a> &#xa0; | &#xa0; -->
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0;  &#xa0;
</p>

<br>

## About ##

This is a simple chatbot powered by OpenAI / HuggingFace LLM models, LangChain framework, FastAPI, Redis, and Google Speech Recognition API. It utlizes LangChain's ReAct Agent to enable LLM based chat and store previous conversations results.


## Technologies ##

The following tools were used in this project:

- [Python](https://www.python.org/)
- [Redis](https://redis.io/)
- [FastAPI](https://fastapi.tiangolo.com/lo/)
- [OpenAI](https://openai.com/)
- [HuggingFace](https://huggingface.co/)

## Requirements ##

Before starting, you need to have [Git](https://git-scm.com), [Docker](https://www.docker.com/) and [Python](https://www.python.org/) installed.

## How To Run Local Server ##

```bash
# Clone this project
$ git clone https://github.com/technickel08/llm-test.git

# Access
$ cd llm-test

# Build Docker 
$ docker build -t langchain-bot .

# Run the project
$ docker run --env OPENAI_API_KEY=<your_key_here> -p 8555:8080 langchain-bot

# The server will initialize in the <http://localhost:8555>
```

&#xa0;

<a href="#top">Back to top</a>
