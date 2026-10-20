import re
import streamlit as st

try:
    import speech_recognition as sr
except ImportError:
    sr = None
from text_analyzer import TextAnalyzer

st.set_page_config(page_title='CodeMix AI', page_icon='🪷', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:linear-gradient(135deg,#f7f7ff,#f3fbff 55%,#fff8f3)}
.block-container{max-width:1280px;padding-top:1.4rem}.hero{padding:1.7rem 1.8rem;border-radius:24px;color:white;background:linear-gradient(120deg,#4338ca,#7c3aed 55%,#db2777);box-shadow:0 14px 38px #5145cd22;margin-bottom:1rem}
.hero h1{color:white!important;font-family:Manrope,sans-serif;font-size:2.4rem;margin:.25rem 0}.hero p{color:#f3efff;margin:0}.panel{background:#fff;border:1px solid #e8e8f4;border-radius:17px;padding:1rem 1.1rem;margin:.3rem 0 .8rem;box-shadow:0 5px 18px #30305a08}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e8e8f4;padding:.8rem;border-radius:15px}div.stButton>button{border-radius:12px;font-weight:700;border:1px solid #dedcf6}.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0}
</style><div class="hero"><div style="letter-spacing:.15em;font-size:.75rem;font-weight:800">🪷 MULTILINGUAL NLP PLAYGROUND</div><h1>CodeMix AI</h1><p>Understand the feeling, context, language mix and tone behind Kannada–English text.</p></div>''', unsafe_allow_html=True)

POS = {'super','sakkath','sakath','sakkat','bombat','bombaat','mast','awesome','excellent','amazing','fantastic','superb','brilliant','chennagide','chennagittu','chanagide','khushi','ಸಕ್ಕತ್','ಸಖತ್','ಬೊಂಬಾಟ್','ಮಸ್ತ್','ಚೆನ್ನಾಗಿದೆ','ಚೆನ್ನಾಗಿತ್ತು','ಅದ್ಭುತ','ಅತ್ಯುತ್ತಮ','ಸೂಪರ್','ಇಷ್ಟವಾಯಿತು','ಖುಷಿ'}
NEG = {'dabba','galiju','galeej','asayya','asahya','holasu','ketta','kettadu','chennagilla','chanagilla','worst','waste','useless','terrible','awful','horrible','pathetic','bekaar','boring','disgusting','mosam','sucks','hate','bad','ಡಬ್ಬಾ','ಗಲೀಜು','ಹೊಲಸು','ಅಸಹ್ಯ','ಕೆಟ್ಟ','ಚೆನ್ನಾಗಿಲ್ಲ','ಕಳಪೆ','ಹಾಳಾಗಿದೆ','ಮೋಸ','ಕಿರಿಕಿರಿ','ನಿಷ್ಪ್ರಯೋಜಕ'}
TOXIC = {'idiot','stupid','fool','moron','shut up','useless fellow','bastard','bitch','asshole','ಮೂರ್ಖ','ನಾಯಿ','ಹುಚ್ಚ','ಕತ್ತೆ'}
EMOTIONS = {
 '😍 Joy / excitement': {'happy','love','awesome','super','sakkath','khushi','ಸಂತೋಷ','ಖುಷಿ','ಚೆನ್ನಾಗಿದೆ','ಇಷ್ಟ'},
 '😠 Anger / frustration': {'angry','hate','irritating','annoying','worst','ಸಿಟ್ಟು','ಕೋಪ','ಕಿರಿಕಿರಿ'},
 '😔 Sadness / disappointment': {'sad','disappointed','upset','ಬೇಸರ','ದುಃಖ','ಕೆಟ್ಟ','ಚೆನ್ನಾಗಿಲ್ಲ'},
 '😲 Surprise': {'wow','omg','surprised','ಅಚ್ಚರಿ','ಅದ್ಭುತ'},
 '😐 Calm / neutral': {'okay','ok','normal','average','ಸರಿ','ಸಾಧಾರಣ','ಪರವಾಗಿಲ್ಲ'}}
EXAMPLES = {
 '😍 Positive slang':'Movie sakkath ide, super acting!',
 '😡 Negative slang':'Ee phone full dabba, chennagilla.',
 '🤔 Mixed feelings':'Acting super, but story tumba boring.',
 '🪷 Kannada script':'ಈ ಮೊಬೈಲ್ ಚೆನ್ನಾಗಿಲ್ಲ, ತುಂಬಾ ಕೆಟ್ಟದಾಗಿದೆ.',
 '💬 Romanized Kannada':'Naanu ee movie tumba ishta patte, songs chennagide.',
 '😐 Neutral':'Phone okay ide, average performance.'}

def cues(text, vocab):
    low=text.lower(); found=[]
    for w in sorted(vocab,key=len,reverse=True):
        if re.search(r'[\u0C80-\u0CFF]',w):
            if w in text: found.append(w)
        elif re.search(r'(?<![a-z])'+re.escape(w)+r'(?![a-z])',low): found.append(w)
    return list(dict.fromkeys(found))

ROMAN_KANNADA_CUES = {
    "naanu", "nanu", "neenu", "enu", "yen", "ide", "illa", "tumba",
    "chennagide", "chennagilla", "sakkath", "maga", "guru", "andre",
    "ishta", "beku", "alla", "aagide", "agide", "maadi", "madthini",
    "yake", "yaake", "ivattu", "nodu", "aythu", "aytu", "thumba",
    "ella", "hege", "hogi", "barutte", "baralla", "phone-u", "madidini",
    "madde", "maadthiya", "gotthu", "gottu", "illa", "andre", "enu",
    "yenu", "hegide", "hegide", "chennagilla", "bekagide", "beda",
    "kooda", "jothe", "mele", "inda", "alli", "illandre", "yavaga",
    "yaavaga", "ivaga", "ivaga", "nodide", "nodbeku", "helidru",
    "helthini", "barbeku", "hogbeku", "maathu", "mathu", "swalpa",
    "svalpa", "thumba", "tumba", "full", "maga", "guru", "ri",
}

def script_profile(text):
    """Report observable script and matched cues, not misleading language percentages."""
    has_kannada = bool(re.search(r'[\u0C80-\u0CFF]', text))
    has_latin = bool(re.search(r'[A-Za-z]', text))
    latin_tokens = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text.lower())
    matched = list(dict.fromkeys(
        token for token in latin_tokens
        if token in ROMAN_KANNADA_CUES
    ))
    if has_kannada and has_latin:
        script = "Mixed scripts (Kannada + Latin)"
    elif has_kannada:
        script = "Kannada script detected"
    elif has_latin:
        script = "Latin script detected"
    else:
        script = "No Kannada/Latin script detected"
    if matched:
        interpretation = "Possible Romanized-Kannada cues found: " + ", ".join(matched)
    elif has_latin:
        interpretation = "Latin-script text detected, but the cue list did not confidently identify Romanized Kannada."
    else:
        interpretation = "No Romanized-Kannada cue check needed for this script."
    return script, matched, interpretation

def emotion_estimate(text):
    scores={name:len(cues(text,words)) for name,words in EMOTIONS.items()}
    joy=scores['😍 Joy / excitement']; neg=scores['😠 Anger / frustration']+scores['😔 Sadness / disappointment']
    if joy and neg: return '😊 + 😞 Mixed emotion'
    best=max(scores,key=scores.get)
    return best if scores[best] else '🙂 Unclear / not detected'

def explain(text, context):
    pos,neg=cues(text,POS),cues(text,NEG)
    if pos and neg: msg='Both positive and negative cue words appear; the sentence may express mixed feelings.'
    elif neg: msg='Negative-looking cue(s) found: '+', '.join(neg)+'. These may have influenced the result.'
    elif pos: msg='Positive-looking cue(s) found: '+', '.join(pos)+'. These may have influenced the result.'
    else: msg='No cue from the small explanation lexicon was found. The label mainly comes from the saved TF-IDF classifier; this is not a token-level explanation.'
    if isinstance(context,dict) and context.get('contextual_meaning'): msg+=' Context helper: '+str(context['contextual_meaning'])
    return msg,pos,neg


def translate_with_fallback(text, target):
    """Try GoogleTranslator, then MyMemory, with explicit diagnostics."""
    errors = []
    source_script_is_kannada = bool(re.search(r"[\u0C80-\u0CFF]", text))
    source = "kn" if source_script_is_kannada else "en"
    # For Latin-script code-mixed text, Google auto-detection is more appropriate.
    try:
        from deep_translator import GoogleTranslator
        google_source = "kn" if source_script_is_kannada else "auto"
        result = GoogleTranslator(source=google_source, target=target).translate(text)
        if result and result.strip():
            return result, "Google Translate"
        errors.append("Google Translate returned an empty result")
    except Exception as exc:
        errors.append(f"Google Translate: {exc}")

    try:
        import json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        # MyMemory language pairs must be explicit. English is the best
        # available default for Latin-script input; mixed Kannada slang may
        # not be understood by this fallback.
        pair_source = "kn" if source_script_is_kannada else "en"
        params = urlencode({"q": text, "langpair": f"{pair_source}|{target}"})
        request = Request(
            "https://api.mymemory.translated.net/get?" + params,
            headers={"User-Agent": "CodeMixAI/1.0"},
        )
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload.get("responseData", {}).get("translatedText", "")
        status = payload.get("responseStatus")
        if status == 200 and result and result.strip():
            return result, "MyMemory"
        detail = payload.get("responseDetails") or f"response status {status}"
        errors.append(f"MyMemory: {detail}")
    except Exception as exc:
        errors.append(f"MyMemory: {exc}")
    return None, "Translation failed. " + " | ".join(errors)


COMMON_ENGLISH_WORDS = {
    "where", "are", "you", "what", "when", "why", "who", "how", "is", "am",
    "the", "a", "an", "this", "that", "these", "those", "hello", "please",
    "thank", "thanks", "good", "morning", "night", "i", "we", "they", "he",
    "she", "it", "my", "your", "our", "their", "can", "could", "would",
    "should", "do", "does", "did", "have", "has", "had", "to", "from", "in",
    "on", "at", "and", "or", "but", "not", "yes", "no", "okay", "ok",
}

def looks_like_plain_english(text):
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return False
    matches = sum(word in COMMON_ENGLISH_WORDS for word in words)
    return matches >= 2 and matches / len(words) >= 0.5

@st.cache_resource(show_spinner='Loading sentiment analyzer…')
def load_analyzer(): return TextAnalyzer()
try: analyzer=load_analyzer()
except Exception as e: st.error(f'Could not load the NLP analyzer: {e}'); st.stop()

with st.sidebar:
    st.markdown('## 🪷 CodeMix AI')
    st.caption('Kannada + English • Sentiment • Context')
    page=st.radio('Workspace',['✨ Text playground','🎙️ Voice analysis','🌐 Translate & transliterate','🧪 About the analysis'],label_visibility='collapsed')
    st.divider(); st.caption('Sentiment uses your saved classifier and cue rules. Emotion, language mix and civility are heuristic estimates.')

def render_analysis(text):
    if not text.strip(): st.warning('Enter some text first.'); return
    try:
        sentiment=str(analyzer.predict_sentiment(text)); topic=analyzer.predict_topic(text); context=analyzer.explain_context(text)
    except Exception as e: st.error(f'Analysis failed: {e}'); return
    low=sentiment.lower(); face='😍' if 'positive' in low else '😞' if 'negative' in low else '😊😞' if 'mixed' in low else '😐'
    why,pos,neg=explain(text,context); emotion=emotion_estimate(text); script,roman_cues,script_note=script_profile(text); toxic=cues(text,TOXIC)
    st.markdown('### 🧠 Analysis at a glance')
    a,b,c=st.columns(3)
    a.markdown(f'<div class="panel" style="background:{"#dcfce7" if "positive" in low else "#fee2e2" if "negative" in low else "#fef3c7" if "mixed" in low else "#e0e7ff"}"><small>SENTIMENT</small><h2>{face} {sentiment}</h2><small>Saved model + cue rules</small></div>',unsafe_allow_html=True)
    b.markdown(f'<div class="panel"><small>EMOTION FINGERPRINT</small><h2>{emotion}</h2><small>Lightweight word-cue estimate</small></div>',unsafe_allow_html=True)
    c.markdown(f'<div class="panel"><small>TOPIC</small><h2>🏷️ {topic}</h2><small>Keyword-based topic helper</small></div>',unsafe_allow_html=True)
    left,right=st.columns([1.1,.9],gap='large')
    with left:
        st.markdown('#### 💡 Why did the AI think that?'); st.write(why)
        if pos: st.markdown('🟢 **Positive cues:** '+', '.join(pos))
        if neg: st.markdown('🔴 **Negative cues:** '+', '.join(neg))
    with right:
        st.markdown('#### 🌐 Script & Romanized-Kannada cues')
        script, roman_cues, script_note = script_profile(text)
        st.write(f'**Script:** {script}')
        st.write(script_note)
        if roman_cues:
            st.markdown('**Matched cues:** ' + ', '.join(f'`{w}`' for w in roman_cues))
        st.caption('No percentages are shown because Romanized Kannada and English overlap heavily in Latin script; a small word list cannot reliably calculate language proportions.')
    st.markdown('#### 🫶 Civility / toxicity check')
    if toxic: st.warning('Potentially harsh terms detected: '+', '.join(toxic))
    else: st.success('No terms from the small civility keyword list were detected.')
    st.caption('A keyword screen can miss harmful wording or flag harmless context; no detection does not guarantee non-toxicity.')
    with st.expander('📖 View the civility / toxicity keyword list'):
        st.write('The app currently checks these exact terms/phrases. The list is intentionally limited and should not be treated as a complete toxicity dictionary.')
        st.code(', '.join(sorted(TOXIC)), language=None)
        st.caption('A term may be harmless in a quoted sentence or used differently by context. This checker only flags matches; it does not determine intent.')
    with st.expander('📚 Contextual meaning & details'):
        st.write(context.get('contextual_meaning','No contextual explanation returned.') if isinstance(context,dict) else str(context))
        if isinstance(context,dict) and context.get('detected_cues'): st.write('**Detected phrase cues:**',', '.join(map(str,context['detected_cues'])))
        st.write('**Script profile:**', script); st.write(script_note); st.write('**Romanized-Kannada cues:**', roman_cues or 'None detected')

if page=='✨ Text playground':
    st.markdown('## ✨ Try a sentence'); st.write('Choose an example or type your own sentence, then explore the results.')
    cols=st.columns(3)
    for i,(label,value) in enumerate(EXAMPLES.items()):
        if cols[i%3].button(label,key=f'ex_{i}',use_container_width=True): st.session_state['cm_text']=value
    text=st.text_area('Kannada-English text',key='cm_text',height=125,placeholder='Example: Movie sakkath ide, but climax tumba boring.')
    if st.button('🔍 Analyze text',type='primary',use_container_width=False): st.session_state['last_text']=text
    if st.session_state.get('last_text') is not None: render_analysis(st.session_state['last_text'])
    with st.expander('🔁 Optional: compare sentence similarity'):
        st.caption('This may take longer if the semantic embedding model needs to load.')
        x=st.text_input('Sentence A',placeholder='Naanu ee movie tumba ishta patte',key='sim_a'); y=st.text_input('Sentence B',placeholder='I really liked this movie',key='sim_b')
        if st.button('Compare meaning similarity'):
            if not x.strip() or not y.strip(): st.warning('Enter both sentences.')
            else:
                try: st.metric('Cosine similarity',f'{analyzer.calculate_similarity(x,y):.3f}'); st.caption('Not a probability or sentiment score.')
                except Exception as e: st.error(f'Similarity comparison failed: {e}')
elif page=='🎙️ Voice analysis':
    st.markdown('## 🎙️ Speak naturally'); st.write('Record, transcribe, review the words, then analyse the transcript.')
    if sr is None: st.error('SpeechRecognition is not installed.')
    else:
        audio=st.audio_input('Record your sentence')
        if audio and st.button('📝 Convert speech to text',type='primary'):
            try:
                rec=sr.Recognizer()
                with sr.AudioFile(audio) as source: data=rec.record(source)
                with st.spinner('Recognizing speech online…'): st.session_state['voice_text']=rec.recognize_google(data,language='kn-IN')
                st.success('Transcript ready. Please review it before analysis.')
            except sr.UnknownValueError: st.error('Speech was unclear. Try speaking more slowly in a quieter place.')
            except sr.RequestError as e: st.error(f'Speech service/network error: {e}')
            except Exception as e: st.error(f'Voice processing failed: {e}')
        transcript=st.text_area('Review or edit transcript',key='voice_text',height=100)
        if st.button('🔍 Analyse transcription',type='primary'): st.session_state['voice_result']=transcript
        if st.session_state.get('voice_result') is not None: render_analysis(st.session_state['voice_result'])
elif page=='🌐 Translate & transliterate':
    st.markdown('## 🌐 Language tools'); st.info('Translation requires internet. Both tools are optional and load only when clicked; install their package if needed.')
    text=st.text_area('Text to convert',height=120,placeholder='ನೀವು ಹೇಗಿದ್ದೀರಿ? / Nivu hegiddiri?')
    t1,t2=st.tabs(['🔤 Translation','🔡 Transliteration'])
    with t1:
        target=st.radio('Translate into',['English','Kannada'],horizontal=True)
        if st.button('🌍 Translate',type='primary'):
            if not text.strip(): st.warning('Enter text to translate.')
            else:
                with st.spinner('Trying online translation services…'):
                    result, provider = translate_with_fallback(text, 'en' if target == 'English' else 'kn')
                if result:
                    st.success(f'Translation received via {provider}.')
                    st.text_area('Translation result', value=result, height=100)
                    st.caption('Machine translation may lose slang, sarcasm and code-mixed meaning.')
                else:
                    st.warning(provider)
    with t2:
        direction = st.radio(
            'Input text format',
            ['Kannada script → Roman (IAST)', 'Roman (IAST) → Kannada script'],
            horizontal=False,
        )
        st.caption('Important: colloquial Kanglish such as “naanu tumba chennagide” is not standardized IAST. This tool transliterates scripts; it does not reliably convert arbitrary Kanglish spellings back to Kannada.')
        if st.button('🔡 Transliterate'):
            if not text.strip():
                st.warning('Enter text to transliterate.')
            else:
                try:
                    from aksharamukha import transliterate
                    if direction.startswith('Kannada'):
                        if not re.search(r'[\u0C80-\u0CFF]', text):
                            st.warning('This direction expects Kannada-script input. Choose the Roman (IAST) → Kannada option for Romanized Kannada.')
                        else:
                            result = transliterate.process('Kannada', 'IAST', text)
                            st.text_area('Roman (IAST) result', value=result, height=100)
                    else:
                        if re.search(r'[\u0C80-\u0CFF]', text):
                            st.warning('This direction expects Romanized input, not Kannada-script text.')
                        elif looks_like_plain_english(text):
                            st.warning('This looks like ordinary English, not Romanized Kannada. Transliteration would produce misleading Kannada letters. Use the Translation tab to translate “' + text + '” into Kannada.')
                        else:
                            result = transliterate.process('IAST', 'Kannada', text)
                            st.text_area('Kannada-script result', value=result, height=100)
                    st.caption('Transliteration changes script, not meaning. Use Translation for English sentences; transliteration is intended for Kannada-script text or standardized Roman (IAST) input.')
                except ImportError:
                    st.warning('Install the optional package in your project terminal: pip install aksharamukha')
                except Exception as e:
                    st.warning(f'Transliteration failed for this input: {e}')
else:
    st.markdown('## 🧪 How to interpret the results')
    st.markdown('''<div class="panel"><h3>🧠 Sentiment</h3>Saved combined word/character TF-IDF classifier plus lightweight phrase-cue overrides. It can struggle with sarcasm, negation and unfamiliar spellings.</div><div class="panel"><h3>😊 Emotion fingerprint</h3>A small word list estimates visible emotional cues; it is not a trained emotion model.</div><div class="panel"><h3>🌐 Language mix</h3>Shows the detected writing script and any matches from a small Romanized-Kannada cue list. It deliberately does not report language percentages because those estimates would be unreliable.</div><div class="panel"><h3>🫶 Civility checker</h3>A limited keyword screen. Open the civility checker’s keyword-list expander to inspect the terms; it is not a robust toxicity classifier or an assessment of intent.</div><div class="panel"><h3>🏷️ Topic and context</h3>Keyword-based helpers; explanations can be generic or incomplete.</div><div class="panel"><h3>🎙️ Voice and 🌍 translation</h3>Depend on online services and network availability. Always review transcripts and translations.</div>''',unsafe_allow_html=True)
    st.caption('Educational project demo only. Do not use heuristic outputs for high-stakes decisions.')
