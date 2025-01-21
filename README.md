<h1>Twitter Sentiment Analysis</h1>
<p>
This code is a sentiment analysis tool built in Python, utilizing the <code>'cardiffnlp/twitter-roberta-base-sentiment'</code> model from the <strong>Transformers</strong> library. 
It categorizes tweets into one of three sentiment categories:
</p>
<ul>
  <li><strong>Neutral</strong></li>
  <li><strong>Negative</strong></li>
  <li><strong>Positive</strong></li>
</ul>
<p>
The tool processes raw tweets by standardizing mentions (e.g., replacing <code>@username</code> with <code>@user</code>) and URLs to ensure consistency before passing the text through the model. 
The output is a set of probabilities indicating the likelihood of each sentiment category.
</p>
<p>
This approach leverages state-of-the-art natural language processing techniques to understand and analyze social media sentiment effectively.
</p>
