---
layout: page
permalink: /projects/
title: projects
description: A selection of projects I have done. Most focus around improving deep learning runtime or action recognition.
---

<ul class="post-list">
{% for project in site.projects reversed %}
    <li>
        <h2><a class="project-title" href="{{ project.url | prepend: site.baseurl }}">{{ project.title }}</a></h2>
        <p class="post-meta">{{ project.date | date: '%B %-d, %Y — %H:%M' }}</p>
      </li>
{% endfor %}
</ul>