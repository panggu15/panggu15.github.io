---
title: "시계열 예측"           <!-- 수정 -->
layout: archive
permalink: categories/시계열예측      <!-- 수정 -->
author_profile: true
sidebar_main: true
---


{% assign posts = site.categories.시계열예측 %}   <!-- 수정 -->
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
