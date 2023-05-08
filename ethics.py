"""
Please answer the following ethics and reflection questions. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

It is possible for users of our chatbot to associate human characteristics to it.
This is because we have designed the chatbot's responses to user input to mimic human
responses, and sympathize with the user. Therefore, users could potentially attribute human
characteristics to our chatbot. One ramification of this is that the user
could develop unrealistic expectations and emotions towards our chatbot, which can lead to
frustration and disappointment when the chatbot is unable to meet the user's expectations.
To make it easier for a user to easily distinguish the chatbot responses from those of a human,
the chatbot could include disclaimers to notify the users that it is a model (like ChatGPT does),
and use different fonts, colors, or formatting for the chatbot responses to make
them distinct from the user's messages.

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

Our chatbot does not currently store information about the user such as their preferences,
personal information, and data about prior user interaction with our bot.
Because our chatbot does not store any personal information identified with a specific user,
the possibility of a data breach is low. However, a number of modern chatbots do store information about users
on their servers, so that they can provide personalized content. Any private user data must be
collected with explicit approval from the user, and steps should be undertaken to
implement strong data encryption and security protocols to protect user data. Moreover
Users should have complete control over their data, and should be able to delete or modify it at any time.

"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of 
tasks that are currently done by paid human workers, such as responding to customer-service queries, 
translating documents or writing computer code. These could displace workers and lead to widespread unemployment. 
What do you think different stakeholders -- elected government officials, employees at technology companies, 
citizens -- should do in anticipation of these risks and/or in response to these real-world harms? 
"""

Q3_your_answer = """

Advances made in modern dialogue systems have huge social costs and benefits.
However, different stakeholders such as employees of the companies making these products or
citizens do not have much input into creating these products.
The first step must be making sure that these stakeholders have a seat in the
descision making process about how these tools are created and deployed.
There should also be a way to redistribute some of the gains (and profits) from the
deployment of these technologies to the citizens so that the social costs of massive
labour displacement in customer-service, translation and programming industry can be mitigated.
Elected government officials should make sure that that these technologies are not
deployed in a way that threatens social order and provide training or re-skilling
programs for workers whose jobs are affected by automation.

"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! What are the advantages and disadvantages of this paradigm 
compared to an end-to-end deep learning approach, e.g. ChatGPT? 
"""

Q4_your_answer = """

One advantage of building a frame-based model such as ours is the modularity and ability
to use different machine learning techniques for specific tasks. However one limitation of this model
compared to a chat-ChatGPT type model is that it is not very genralizable or flexible and
can not properly handle input/tasks that it has not been explicitly programmed for.
Building these modular pieces is also very time-consuming since a different technique
is used for each task, compared to a bigger neural network model where a uniform bigger base
model is used to mini-tasks. 

"""