---
title: "기초 데이터 분석"
layout: archive
permalink: categories/basic
author_profile: true
sidebar_main: true
---


{% assign posts = site.categories.basic %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}