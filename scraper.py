# scraper
from typing import List
import json
    
import requests
from bs4 import BeautifulSoup

verboseoutput = False
showoutput = False
outputfile = "questions.json"
outputansfile = "answers.json"

def scrapeQuestionsTitles(url: str = "https://github.com/csujedihy/lc-all-solutions", v: bool = False) -> List:
    '''
    scrapes the names of the questions from https://github.com/csujedihy/lc-all-solutions
    '''
    questions = []
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        questionelements = soup.find_all('a', class_='Link--primary')

        for q in questionelements:
            if q.text.strip()[0].isdigit():
                if v: print(q.text.strip())
                questions.append(q.text.strip())

        # removes duplicates
        questions = list(set(questions))
        questions.sort()

        return questions
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def titlesToQnUrl(titles: List, v: bool = False) -> List:
    '''
    converts the titles of the questions to their respective urls
    '''
    urls = []
    for title in titles:
        url = f"https://raw.githubusercontent.com/csujedihy/lc-all-solutions/refs/heads/master/{title}/question.md"
        if v: print(url)
        urls.append(url)
    return urls

def titlesToAnsUrl(titles: List, v: bool = False) -> List:
    '''
    converts the titles of the answers to their respective urls
    '''
    urls = []
    for title in titles:
        url = f"https://raw.githubusercontent.com/csujedihy/lc-all-solutions/refs/heads/master/{title}/{title[4:]}.py"
        if v: print(url)
        urls.append(url)
    return urls

def scrapeQuestions(urls: List, v: bool = False) -> List:
    qnList = []
    errList = []
    '''
    scrapes the questions from https://github.com/csujedihy/lc-all-solutions
    '''
    for questionurl in urls:
        try:
            response = requests.get(questionurl)
            response.raise_for_status()
            if v: print(response.text)

            qnList.append(response.text)
        except Exception as e:
            print(f"An error occurred: {e}\nurl: {questionurl}")
            errList.append(questionurl)
        
    return qnList, errList

def scrapeAnswers(urls: List, v: bool = False) -> List:
    ansList = []
    errList = []
    '''
    scrapes the questions from https://github.com/csujedihy/lc-all-solutions
    '''
    for answerurl in urls:
        try:
            response = requests.get(answerurl)
            response.raise_for_status()
            if v: print(response.text)

            ansList.append(response.text)
        except Exception as e:
            print(f"An error occurred: {e}\nurl: {answerurl}")
            errList.append(answerurl)
        
    return ansList, errList

print("scraping question titles...", end="") 
questionTitles = scrapeQuestionsTitles(v = verboseoutput)
print(".................done")
print()

print("converting question titles to urls...", end="")
urls = titlesToQnUrl(questionTitles, verboseoutput)
print(".......done")
print()

print("scraping questions...", end="")
questions, errors = scrapeQuestions(urls, verboseoutput)
print(".......................done")
print()

print("converting answer titles to answer urls...", end="")
ansurls = titlesToAnsUrl(questionTitles, verboseoutput)
print("..done")
print()

print("scraping answers...", end="")
answers, errors = scrapeAnswers(ansurls, verboseoutput)
print(".........................done")
print()

if verboseoutput or showoutput:
    print(f"questions: {questions}")
    print(f"qrrors: {errors}")

dqns = {}
dans = {}

dqns = {i: question for i, question in enumerate(questions)}
dans = {i: answer for i, answer in enumerate(answers)}

print(f"writing questions to file...", end="")
with open(outputfile, "w") as f:
    json.dump(dqns, f)
print("................done")
print()

print(f"writing answers to file...", end="")
with open(outputansfile, "w") as f:
    json.dump(dans, f)
print("..................done")
print()

print(f"scraped questions are stored in {outputfile} or as the variable questions")