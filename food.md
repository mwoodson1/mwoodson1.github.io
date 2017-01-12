---
layout: page
title: food
permalink: /food/
---
Take a look at some of my own, and others, culinary creations! I do not believe in recipes if you truly want to cook but I'll try to provide all ingredients and tips in making the things I made. If you still want a recipe shoot me a mail and I can make up some numbers for you :).

{% for food in site.food %}

{% if food.redirect %}
<div class="project">
    <div class="thumbnail">
        <a href="{{ food.redirect }}" target="_blank">
        {% if food.img %}
        <img class="thumbnail" src="{{ food.img }}"/>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}
        <span>
            <h1>{{ food.title }}</h1>
            <br/>
            <p>{{ food.description }}</p>
        </span>
        </a>
    </div>
</div>
{% else %}

<div class="project ">
    <div class="thumbnail">
        <a href="{{ site.baseurl }}{{ food.url }}">
        {% if food.img %}
        <img class="thumbnail" src="{{ food.img }}"/>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}
        <span>
            <h3>{{ food.title }}</h3>
            <br/>
            <p>{{ food.description }}</p>
        </span>
        </a>
    </div>
</div>

{% endif %}

{% endfor %}
