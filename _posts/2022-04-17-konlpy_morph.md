---
layout: single
title:  "[파이썬 konlpy] 한국어 형태소 분석 및 비교"
categories: nlp_anal
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# konlpy



형태소 분석: 실제 문장에서 사용되는 단어의 원래 구조를 파악하는 것



KoNLPy패키지에는 Hannanum, Kkma, Komoran, Mecab, Okt(Twitter)등의 클래스들이 있습니다



이 클래스들을 이용해 명사와 품사 태그들을 반환합니다.



morphs(형태소 반환), nouns(명사 반환), pos(품사정보 반환) 메소드가 공통으로 존재하고 형태소 분석기에 따라 다른 기능이 있습니다.



```python
!pip install konlpy
```

# Hannanum(한나눔)



KAIST SWRC(Semantic Web Research Center)에서 개발



- morphs: 텍스트에서 형태소를 반환



- nouns: 텍스트에서 명사를 반환



- pos: 텍스트에서 품사 정보를 부착하여 반환



- tagset: 품사 정보의 뜻 반환



```python
from konlpy.tag import Hannanum
 
hannanum = Hannanum()
```


```python
hannanum.morphs("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어',
 '는',
 '주변',
 '언어',
 '와',
 '어떤',
 '친족',
 '관계',
 '도',
 '밝혀지',
 '지',
 '않',
 '은',
 '언어',
 '이',
 '다']
</pre>

```python
hannanum.nouns("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어', '주변', '언어', '친족', '관계', '언어']
</pre>

```python
hannanum.pos("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
[('한국어', 'N'),
 ('는', 'J'),
 ('주변', 'N'),
 ('언어', 'N'),
 ('와', 'J'),
 ('어떤', 'M'),
 ('친족', 'N'),
 ('관계', 'N'),
 ('도', 'J'),
 ('밝혀지', 'P'),
 ('지', 'E'),
 ('않', 'P'),
 ('은', 'E'),
 ('언어', 'N'),
 ('이', 'J'),
 ('다', 'E')]
</pre>

```python
hannanum.tagset
```

<pre>
{'E': '어미',
 'EC': '연결 어미',
 'EF': '종결 어미',
 'EP': '선어말어미',
 'ET': '전성 어미',
 'F': '외국어',
 'I': '독립언',
 'II': '감탄사',
 'J': '관계언',
 'JC': '격조사',
 'JP': '서술격 조사',
 'JX': '보조사',
 'M': '수식언',
 'MA': '부사',
 'MM': '관형사',
 'N': '체언',
 'NB': '의존명사',
 'NC': '보통명사',
 'NN': '수사',
 'NP': '대명사',
 'NQ': '고유명사',
 'P': '용언',
 'PA': '형용사',
 'PV': '동사',
 'PX': '보조 용언',
 'S': '기호',
 'X': '접사',
 'XP': '접두사',
 'XS': '접미사'}
</pre>
# Kkma(꼬꼬마)



서울대학교 IDS(intelligent Data Systems) 연구실에서 개발



- morphs: 텍스트에서 형태소를 반환



- nouns: 텍스트에서 명사를 반환



- pos: 텍스트에서 품사 정보를 부착하여 반환



- tagset: 품사 정보의 뜻 반환



- sentences: 문장별로 반환




```python
from konlpy.tag import Kkma

kkma = Kkma()
```


```python
kkma.morphs("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어',
 '는',
 '주변',
 '언어',
 '와',
 '어떻',
 'ㄴ',
 '친족',
 '관계',
 '도',
 '밝혀지',
 '지',
 '않',
 '은',
 '언어',
 '이',
 '다']
</pre>

```python
kkma.nouns("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어', '주변', '언어', '친족', '관계']
</pre>

```python
kkma.pos("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
[('한국어', 'NNG'),
 ('는', 'JX'),
 ('주변', 'NNG'),
 ('언어', 'NNG'),
 ('와', 'JKM'),
 ('어떻', 'VA'),
 ('ㄴ', 'ETD'),
 ('친족', 'NNG'),
 ('관계', 'NNG'),
 ('도', 'JX'),
 ('밝혀지', 'VV'),
 ('지', 'ECD'),
 ('않', 'VXV'),
 ('은', 'ETD'),
 ('언어', 'NNG'),
 ('이', 'VCP'),
 ('다', 'EFN')]
</pre>

```python
kkma.tagset
```

<pre>
{'EC': '연결 어미',
 'ECD': '의존적 연결 어미',
 'ECE': '대등 연결 어미',
 'ECS': '보조적 연결 어미',
 'EF': '종결 어미',
 'EFA': '청유형 종결 어미',
 'EFI': '감탄형 종결 어미',
 'EFN': '평서형 종결 어미',
 'EFO': '명령형 종결 어미',
 'EFQ': '의문형 종결 어미',
 'EFR': '존칭형 종결 어미',
 'EP': '선어말 어미',
 'EPH': '존칭 선어말 어미',
 'EPP': '공손 선어말 어미',
 'EPT': '시제 선어말 어미',
 'ET': '전성 어미',
 'ETD': '관형형 전성 어미',
 'ETN': '명사형 전성 어미',
 'IC': '감탄사',
 'JC': '접속 조사',
 'JK': '조사',
 'JKC': '보격 조사',
 'JKG': '관형격 조사',
 'JKI': '호격 조사',
 'JKM': '부사격 조사',
 'JKO': '목적격 조사',
 'JKQ': '인용격 조사',
 'JKS': '주격 조사',
 'JX': '보조사',
 'MA': '부사',
 'MAC': '접속 부사',
 'MAG': '일반 부사',
 'MD': '관형사',
 'MDN': '수 관형사',
 'MDT': '일반 관형사',
 'NN': '명사',
 'NNB': '일반 의존 명사',
 'NNG': '보통명사',
 'NNM': '단위 의존 명사',
 'NNP': '고유명사',
 'NP': '대명사',
 'NR': '수사',
 'OH': '한자',
 'OL': '외국어',
 'ON': '숫자',
 'SE': '줄임표',
 'SF': '마침표, 물음표, 느낌표',
 'SO': '붙임표(물결,숨김,빠짐)',
 'SP': '쉼표,가운뎃점,콜론,빗금',
 'SS': '따옴표,괄호표,줄표',
 'SW': '기타기호 (논리수학기호,화폐기호)',
 'UN': '명사추정범주',
 'VA': '형용사',
 'VC': '지정사',
 'VCN': "부정 지정사, 형용사 '아니다'",
 'VCP': "긍정 지정사, 서술격 조사 '이다'",
 'VV': '동사',
 'VX': '보조 용언',
 'VXA': '보조 형용사',
 'VXV': '보조 동사',
 'XP': '접두사',
 'XPN': '체언 접두사',
 'XPV': '용언 접두사',
 'XR': '어근',
 'XSA': '형용사 파생 접미사',
 'XSN': '명사파생 접미사',
 'XSV': '동사 파생 접미사'}
</pre>

```python
kkma.sentences('드디어 대학 졸업했어 이제 난 백수야')
```

<pre>
['드디어 대학 졸업했어', '이제 난 백수야']
</pre>
# Komoran(코모란)



- morphs: 텍스트에서 형태소를 반환



- nouns: 텍스트에서 명사를 반환



- pos: 텍스트에서 품사 정보를 부착하여 반환



- tagset: 품사 정보의 뜻 반환



```python
from konlpy.tag import Komoran 

komoran = Komoran()
```


```python
komoran.morphs("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어',
 '는',
 '주변',
 '언어',
 '와',
 '어떤',
 '친족',
 '관계',
 '도',
 '밝히',
 '어',
 '지',
 '지',
 '않',
 '은',
 '언어',
 '다']
</pre>

```python
komoran.nouns("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어', '주변', '언어', '친족', '관계', '언어']
</pre>

```python
komoran.pos("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
[('한국어', 'NNP'),
 ('는', 'JX'),
 ('주변', 'NNG'),
 ('언어', 'NNG'),
 ('와', 'JC'),
 ('어떤', 'MM'),
 ('친족', 'NNG'),
 ('관계', 'NNG'),
 ('도', 'JX'),
 ('밝히', 'VV'),
 ('어', 'EC'),
 ('지', 'VX'),
 ('지', 'EC'),
 ('않', 'VX'),
 ('은', 'ETM'),
 ('언어', 'NNG'),
 ('다', 'JX')]
</pre>

```python
komoran.tagset
```

<pre>
{'EC': '연결 어미',
 'EF': '종결 어미',
 'EP': '선어말어미',
 'ETM': '관형형 전성 어미',
 'ETN': '명사형 전성 어미',
 'IC': '감탄사',
 'JC': '접속 조사',
 'JKB': '부사격 조사',
 'JKC': '보격 조사',
 'JKG': '관형격 조사',
 'JKO': '목적격 조사',
 'JKQ': '인용격 조사',
 'JKS': '주격 조사',
 'JKV': '호격 조사',
 'JX': '보조사',
 'MAG': '일반 부사',
 'MAJ': '접속 부사',
 'MM': '관형사',
 'NA': '분석불능범주',
 'NF': '명사추정범주',
 'NNB': '의존 명사',
 'NNG': '일반 명사',
 'NNP': '고유 명사',
 'NP': '대명사',
 'NR': '수사',
 'NV': '용언추정범주',
 'SE': '줄임표',
 'SF': '마침표, 물음표, 느낌표',
 'SH': '한자',
 'SL': '외국어',
 'SN': '숫자',
 'SO': '붙임표(물결,숨김,빠짐)',
 'SP': '쉼표,가운뎃점,콜론,빗금',
 'SS': '따옴표,괄호표,줄표',
 'SW': '기타기호 (논리수학기호,화폐기호)',
 'VA': '형용사',
 'VCN': '부정 지정사',
 'VCP': '긍정 지정사',
 'VV': '동사',
 'VX': '보조 용언',
 'XPN': '체언 접두사',
 'XR': '어근',
 'XSA': '형용사 파생 접미사',
 'XSN': '명사파생 접미사',
 'XSV': '동사 파생 접미사'}
</pre>
# Okt (Twitter)



오픈 소스 한국어 분석기이고, 과거 트위터 형태소 분석기였습니다.



- morphs: 텍스트에서 형태소를 반환



- nouns: 텍스트에서 명사를 반환



- pos: 텍스트에서 품사 정보를 부착하여 반환



- tagset: 품사 정보의 뜻 반환



- phrases: 어절 반환



```python
from konlpy.tag import Okt 

okt = Okt()
```


```python
okt.morphs("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어', '는', '주변', '언어', '와', '어떤', '친족', '관계도', '밝혀지지', '않은', '언어', '다']
</pre>

```python
okt.nouns("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어', '주변', '언어', '친족', '관계도', '언어']
</pre>

```python
okt.pos("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
[('한국어', 'Noun'),
 ('는', 'Josa'),
 ('주변', 'Noun'),
 ('언어', 'Noun'),
 ('와', 'Josa'),
 ('어떤', 'Adjective'),
 ('친족', 'Noun'),
 ('관계도', 'Noun'),
 ('밝혀지지', 'Verb'),
 ('않은', 'Verb'),
 ('언어', 'Noun'),
 ('다', 'Josa')]
</pre>

```python
okt.tagset
```

<pre>
{'Adjective': '형용사',
 'Adverb': '부사',
 'Alpha': '알파벳',
 'Conjunction': '접속사',
 'Determiner': '관형사',
 'Eomi': '어미',
 'Exclamation': '감탄사',
 'Foreign': '외국어, 한자 및 기타기호',
 'Hashtag': '트위터 해쉬태그',
 'Josa': '조사',
 'KoreanParticle': '(ex: ㅋㅋ)',
 'Noun': '명사',
 'Number': '숫자',
 'PreEomi': '선어말어미',
 'Punctuation': '구두점',
 'ScreenName': '트위터 아이디',
 'Suffix': '접미사',
 'Unknown': '미등록어',
 'Verb': '동사'}
</pre>

```python
okt.phrases("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다")
```

<pre>
['한국어', '주변', '주변 언어', '주변 언어와 어떤 친족', '주변 언어와 어떤 친족 관계도', '언어', '친족', '관계도']
</pre>