"""Browser transport: serves the hands-free voice UI (a smiley face whose mouth
animates to the spoken audio) and runs one voice turn per request.

Capture uses a continuous recorder with client-side voice-activity detection
(pre-roll, so the first word is never clipped); it auto-detects when you stop,
sends the utterance, speaks the reply while animating the mouth, then listens
again. No buttons, no text."""
import base64
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context

from . import config
from .tts import sentence_chunks

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aware</title>
<style>
:root{
 --bg:#090b10;--bg-glow:#151c2a;--skin:#f3c94f;--skin-hi:#ffe786;--ink:#17140b;
 --accent:#72d8ff;--active:#62edbd;--think:#b89cff;--speak:#ffc85c;--danger:#ff766e;
 --muted:#8c94a4;--text:#e7ebf2;--voice-glow:0px;--voice-scale:0;
}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:
 radial-gradient(circle at 50% 42%,color-mix(in srgb,var(--bg-glow) 75%,transparent) 0,transparent 44%),
 radial-gradient(circle at 50% 120%,#162031 0,transparent 48%),var(--bg);
 color:var(--text);display:flex;flex-direction:column;align-items:center;justify-content:center;
 min-height:100%;cursor:pointer;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display",Inter,system-ui,sans-serif}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.15;
 background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.14'/%3E%3C/svg%3E")}
.brand{position:fixed;top:clamp(22px,4vh,42px);left:50%;transform:translateX(-50%);font-size:10px;
 font-weight:700;letter-spacing:.32em;text-indent:.32em;color:#727b8e;text-transform:uppercase}
.presence{position:relative;width:min(62vw,62vh);height:min(62vw,62vh);display:grid;place-items:center;
 isolation:isolate;transition:filter .7s ease,opacity .5s ease,transform .7s cubic-bezier(.16,1,.3,1)}
.presence:before,.presence:after{content:"";position:absolute;inset:5%;border-radius:50%;pointer-events:none}
.presence:before{border:1px solid color-mix(in srgb,var(--accent) 18%,transparent);opacity:.38;
 box-shadow:0 0 50px color-mix(in srgb,var(--accent) 7%,transparent),inset 0 0 45px color-mix(in srgb,var(--accent) 4%,transparent);
 transform:scale(.94);transition:all .65s cubic-bezier(.2,.7,.2,1)}
.presence:after{opacity:0;background:conic-gradient(from 0deg,transparent 0 55%,var(--think) 72%,transparent 87%);
 -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 2px),#000 0);mask:radial-gradient(farthest-side,transparent calc(100% - 2px),#000 0)}
#face{position:relative;z-index:2;width:74%;height:74%;overflow:visible;
 filter:drop-shadow(0 24px 34px rgba(0,0,0,.34));transition:opacity .5s,filter .5s,transform .5s}
#face.off{filter:grayscale(.5) drop-shadow(0 18px 30px rgba(0,0,0,.25));opacity:.56}
body[data-state="listening"] .presence:before{border-color:color-mix(in srgb,var(--accent) 45%,transparent);
 box-shadow:0 0 calc(45px + var(--voice-glow)) color-mix(in srgb,var(--accent) 18%,transparent),
 inset 0 0 50px color-mix(in srgb,var(--accent) 7%,transparent);animation:listenPulse 3.2s ease-in-out infinite}
body[data-state="recording"] .presence:before{border-color:color-mix(in srgb,var(--active) 75%,transparent);
 transform:scale(calc(.96 + var(--voice-scale)));box-shadow:0 0 calc(52px + var(--voice-glow)) color-mix(in srgb,var(--active) 24%,transparent)}
body[data-state="thinking"] .presence:before{border-color:color-mix(in srgb,var(--think) 25%,transparent);transform:scale(.9)}
body[data-state="thinking"] .presence:after{opacity:.9;animation:orbit 1.7s linear infinite}
body[data-state="speaking"] .presence:before{border-color:color-mix(in srgb,var(--speak) 50%,transparent);
 transform:scale(calc(.95 + var(--voice-scale)));box-shadow:0 0 calc(50px + var(--voice-glow)) color-mix(in srgb,var(--speak) 18%,transparent)}
body[data-looking="true"] .presence{filter:drop-shadow(0 0 28px rgba(98,237,189,.22))}
body[data-looking="true"] .presence:after{opacity:.55;background:conic-gradient(from 0deg,transparent 0 68%,var(--active) 78%,transparent 89%);animation:orbit 3.5s linear infinite}
body[data-vision="true"] .presence{filter:blur(9px);opacity:.16;transform:scale(.82)}
body[data-vision="true"] .readout{transform:translateY(12px);opacity:.42}
@keyframes listenPulse{0%,100%{transform:scale(.94);opacity:.42}50%{transform:scale(1);opacity:.85}}
@keyframes orbit{to{transform:rotate(360deg)}}
.skin{fill:url(#skinGradient);transition:filter .45s}
#face.looking .skin{filter:hue-rotate(65deg) saturate(.82)}
.face-shade{fill:none;stroke:rgba(255,255,255,.2);stroke-width:1}
.eye{fill:var(--ink);transform-box:fill-box;transform-origin:center;animation:blink 5.4s infinite}
#smile{fill:none;stroke:var(--ink);stroke-width:9;stroke-linecap:round}
#mouth{fill:var(--ink)}
body[data-state="thinking"] #face{animation:thinkBreath 1.7s ease-in-out infinite}
@keyframes thinkBreath{0%,100%{transform:scale(.985)}50%{transform:scale(1.015)}}
@keyframes blink{0%,95%,100%{transform:scaleY(1)}97.5%{transform:scaleY(.08)}}
.readout{display:flex;flex-direction:column;align-items:center;gap:12px;min-height:92px;width:min(84vw,620px);z-index:2;
 transition:opacity .55s ease,transform .65s cubic-bezier(.2,.8,.2,1)}
#mode{display:inline-flex;align-items:center;gap:9px;color:var(--muted);font-size:11px;font-weight:700;
 letter-spacing:.2em;text-indent:.2em;text-transform:uppercase;transition:color .35s}
#modeDot{width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 0 0 currentColor;transition:color .35s}
body[data-state="listening"] #mode{color:var(--accent)}body[data-state="recording"] #mode{color:var(--active)}
body[data-state="thinking"] #mode{color:var(--think)}body[data-state="speaking"] #mode{color:var(--speak)}
body[data-state="listening"] #modeDot,body[data-state="recording"] #modeDot{animation:dotPulse 1.6s ease-out infinite}
@keyframes dotPulse{70%,100%{box-shadow:0 0 0 9px transparent}}
#status{font:450 clamp(14px,1.8vh,17px)/1.55 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;
 color:var(--muted);text-align:center;max-width:100%;min-height:3em;letter-spacing:.005em;
 white-space:pre-wrap;transition:color .3s;cursor:default}
#status.err{color:var(--danger);font-weight:600}#status b{color:var(--text);font-weight:620}
.hint{position:fixed;bottom:clamp(20px,3.5vh,38px);font-size:11px;color:#596171;letter-spacing:.035em;opacity:0;transition:opacity .4s}
body[data-state="idle"] .hint{opacity:1}
.vision-lens{position:fixed;z-index:8;left:50%;top:50%;width:min(78vw,560px);transform:translate(-50%,-46%) scale(.92);
 opacity:0;visibility:hidden;pointer-events:none;transition:opacity .38s ease,transform .65s cubic-bezier(.16,1,.3,1),visibility 0s linear .65s}
body[data-vision="true"] .vision-lens{opacity:1;visibility:visible;transform:translate(-50%,-50%) scale(1);transition-delay:0s}
.vision-head{display:flex;align-items:flex-end;justify-content:space-between;margin:0 3px 13px}
.vision-kicker{color:var(--active);font-size:10px;font-weight:720;letter-spacing:.22em;text-transform:uppercase}
#visionTitle{margin-top:5px;color:var(--text);font-size:clamp(17px,2.5vw,22px);font-weight:570;letter-spacing:-.025em}
.vision-live{display:flex;align-items:center;gap:7px;color:#788294;font-size:10px;font-weight:650;letter-spacing:.12em;text-transform:uppercase}
.vision-live:before{content:"";width:5px;height:5px;border-radius:50%;background:var(--active);box-shadow:0 0 10px rgba(98,237,189,.75);animation:livePulse 1.4s ease-in-out infinite}
@keyframes livePulse{50%{opacity:.35}}
.vision-frame{position:relative;aspect-ratio:4/3;overflow:hidden;border-radius:clamp(18px,3vw,28px);background:#11151d;
 border:1px solid rgba(255,255,255,.11);box-shadow:0 30px 90px rgba(0,0,0,.52),0 0 0 1px rgba(0,0,0,.35)}
#cam{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1);filter:saturate(.72) contrast(1.06) brightness(.8)}
.vision-frame:before{content:"";position:absolute;z-index:2;inset:0;pointer-events:none;background:
 linear-gradient(to bottom,rgba(7,10,15,.06),transparent 56%,rgba(7,10,15,.54)),
 radial-gradient(circle at center,transparent 48%,rgba(5,7,11,.28));box-shadow:inset 0 0 55px rgba(0,0,0,.22)}
.scan-line{position:absolute;z-index:3;left:0;right:0;top:0;height:1px;opacity:.55;
 background:linear-gradient(90deg,transparent,var(--active),transparent);box-shadow:0 0 14px rgba(98,237,189,.5);animation:scan 2.6s ease-in-out infinite}
@keyframes scan{0%{transform:translateY(-2px);opacity:0}12%{opacity:.6}88%{opacity:.6}100%{transform:translateY(calc(min(78vw,560px)*.75));opacity:0}}
#visionMarks{position:absolute;z-index:4;inset:0}
.track{position:absolute;border:1px solid rgba(98,237,189,.78);border-radius:12px;min-width:38px;min-height:38px;
 box-shadow:0 0 0 1px rgba(0,0,0,.12),inset 0 0 22px rgba(98,237,189,.04);transition:left .35s,top .35s,width .35s,height .35s}
.track:before,.track:after{content:"";position:absolute;width:11px;height:11px;border-color:var(--active)}
.track:before{left:-2px;top:-2px;border-left:2px solid;border-top:2px solid;border-radius:4px 0 0}
.track:after{right:-2px;bottom:-2px;border-right:2px solid;border-bottom:2px solid;border-radius:0 0 4px}
.track-label{position:absolute;left:-1px;bottom:-31px;display:flex;align-items:center;gap:8px;white-space:nowrap;
 color:#f2f5f7;font-size:11px;font-weight:620;text-shadow:0 1px 5px rgba(0,0,0,.8)}
.track-label em{font-style:normal;color:#a7b0bd;font-weight:500}
.vision-empty{position:absolute;z-index:4;left:50%;top:50%;transform:translate(-50%,-50%);color:#8b95a3;
 font-size:11px;font-weight:650;letter-spacing:.12em;text-transform:uppercase;opacity:0;transition:opacity .3s}
.vision-empty.show{opacity:1}
.vision-foot{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:13px 3px 0;color:#747e8e;font-size:11px}
#visionSummary{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.vision-private{flex:none;color:#5d6674}
@media (max-width:600px){.vision-lens{width:min(90vw,520px)}.vision-private{display:none}.vision-frame{border-radius:20px}}
@media (max-height:620px){.brand{top:16px}.presence{width:min(55vw,55vh);height:min(55vw,55vh)}.readout{gap:8px;min-height:70px}.hint{display:none}.vision-lens{width:min(62vw,440px)}.vision-head{margin-bottom:8px}.vision-foot{margin-top:8px}}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.001ms!important;animation-iteration-count:1!important;transition-duration:.001ms!important}}
</style></head>
<body data-state="idle" data-looking="false">
 <div class="brand">Aware</div>
 <div class="presence" aria-hidden="true">
  <svg id="face" class="face off" viewBox="0 0 200 200">
   <defs><radialGradient id="skinGradient" cx="38%" cy="28%" r="78%"><stop offset="0" stop-color="var(--skin-hi)"/><stop offset=".58" stop-color="var(--skin)"/><stop offset="1" stop-color="#dcae2e"/></radialGradient></defs>
   <circle class="skin" cx="100" cy="100" r="94"/>
   <circle class="face-shade" cx="100" cy="100" r="92.5"/>
   <circle class="eye" cx="70" cy="84" r="10"/>
   <circle class="eye" cx="130" cy="84" r="10"/>
   <path id="smile" d="M62 122 Q100 156 138 122"/>
   <ellipse id="mouth" cx="100" cy="130" rx="27" ry="3" style="display:none"/>
  </svg>
 </div>
 <div class="readout" aria-live="polite">
  <div id="mode"><span id="modeDot"></span><span id="modeText">Standby</span></div>
  <div id="status">Click the face to begin</div>
 </div>
 <div class="hint">Click the face anytime to pause</div>
 <section id="visionLens" class="vision-lens" aria-hidden="true">
  <div class="vision-head">
   <div><div class="vision-kicker">Visual context</div><div id="visionTitle">Reading the room</div></div>
   <div class="vision-live">Local</div>
  </div>
  <div class="vision-frame">
   <video id="cam" autoplay playsinline muted></video>
   <div class="scan-line"></div><div id="visionMarks"></div><div id="visionEmpty" class="vision-empty">Seeking a face</div>
  </div>
  <div class="vision-foot"><span id="visionSummary">Checking identity and expression</span><span class="vision-private">On-device vision</span></div>
 </section>
 <canvas id="vcanvas" style="display:none"></canvas>
<script>
const faceEl=document.getElementById('face'),smile=document.getElementById('smile'),mouth=document.getElementById('mouth'),
 statusEl=document.getElementById('status'),modeText=document.getElementById('modeText'),
 cam=document.getElementById('cam'),vcanvas=document.getElementById('vcanvas'),
 visionLens=document.getElementById('visionLens'),visionTitle=document.getElementById('visionTitle'),
 visionMarks=document.getElementById('visionMarks'),visionEmpty=document.getElementById('visionEmpty'),
 visionSummary=document.getElementById('visionSummary');
let started=false,handsFree=true,looking=false,baseFace='';
let micStream=null,audioCtx=null,analyser=null,dataArr=null,lastT=0;
let calibrated=false,calibVals=[],noiseFloor=0.01;
let convState='idle',pcmNode=null,silentGain=null,pcmChunks=[],pcmPreroll=[],segStart=0,curSrc=null;
let streamSources=new Set(),streamDone=false,playbackEnd=0,streamAbort=null;
let voicedMs=0,silenceMs=0,speechMs=0,hasSpeech=false,pending=null;
let visionOpen=false,visionKind='',visionBusy=false;
const ENDPOINT_MS=__ENDPOINT_MS__,MIN_SPEECH_MS=240,MAX_UTTER_MS=15000,LISTEN_RESET_MS=8000;

const MODE_LABEL={idle:'Standby',listening:'Listening',recording:'Hearing you',thinking:'Thinking',speaking:'Speaking'};
function face(cls){
 baseFace=cls;const state=cls||'idle';
 faceEl.setAttribute('class','face'+(cls?' '+cls:'')+(looking?' looking':''));
 document.body.dataset.state=state;document.body.dataset.looking=String(looking);
 modeText.textContent=MODE_LABEL[state]||state;
}
// The UI must always say what it is doing. A turn that fails is NEVER swallowed:
// fatal(msg) stops the loop and shows why, instead of silently listening again.
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function say(html,err){statusEl.innerHTML=html;statusEl.classList.toggle('err',!!err)}
function showVision(kind='analysis',enroll=null){
 if(!camOk)return;
 visionOpen=true;visionKind=kind;document.body.dataset.vision='true';visionLens.setAttribute('aria-hidden','false');
 if(kind==='enroll'&&enroll){
   const action=enroll.kind==='emotion'?'Learning '+(enroll.expression||'expression'):'Learning '+(enroll.name||'this face');
   visionTitle.textContent=action;visionSummary.textContent=(enroll.captured||0)+' of '+(enroll.target||0)+' clean views captured';
 }else{visionTitle.textContent='Reading the room';visionSummary.textContent='Checking identity and expression'}
 refreshVision();
}
function hideVision(){
 visionOpen=false;visionKind='';document.body.dataset.vision='false';visionLens.setAttribute('aria-hidden','true');
}
function pct(n){return Math.round(Math.max(0,Math.min(1,Number(n)||0))*100)+'%'}
function renderVision(scene){
 const people=Array.isArray(scene.people)?scene.people:[],fw=scene.frame_width||320,fh=scene.frame_height||240;
 // Normal turns deliberately pause expensive board inference while the LLM runs.
 // Keep the last honest snapshot on screen instead of replacing it with a fake
 // "no face" result when that snapshot ages out.
 if(visionKind==='analysis'&&scene.feed_live===false&&visionMarks.childElementCount){
   visionTitle.textContent='Visual context captured';visionSummary.textContent='Reasoning with the scene snapshot';return;
 }
 visionMarks.replaceChildren();visionEmpty.classList.toggle('show',!people.length);
 for(const p of people){
   const b=p.bbox||[0,0,0,0],mark=document.createElement('div'),label=document.createElement('div');
   // The preview is mirrored, so mirror the detector's x coordinate as well.
   mark.className='track';mark.style.left=(100-(b[0]+b[2])*100/fw)+'%';mark.style.top=(b[1]*100/fh)+'%';
   mark.style.width=(b[2]*100/fw)+'%';mark.style.height=(b[3]*100/fh)+'%';label.className='track-label';
   const name=p.name&&p.name!=='unknown'?p.name:'Unrecognized';
   const emotion=p.emotion&&p.sentiment!=='not_enabled'?p.emotion:'';
   label.innerHTML='<span>'+esc(name)+'</span><em>'+(emotion?esc(emotion)+' · '+pct(p.emotion_score):pct(p.identity_score))+'</em>';
   mark.appendChild(label);visionMarks.appendChild(mark);
 }
 if(visionKind!=='enroll'){
   visionTitle.textContent=people.length?'Recognizing '+people.length+(people.length===1?' person':' people'):'Checking the scene';
   visionSummary.textContent=people.length?people.map(p=>(p.name&&p.name!=='unknown'?p.name:'Unknown')+(p.emotion?' · '+p.emotion:'')).join('   '):'No face currently detected';
 }
}
async function refreshVision(){
 if(!visionOpen||visionBusy)return;visionBusy=true;
 try{const r=await fetch('/api/vision/scene');if(r.ok)renderVision(await r.json())}catch(e){}finally{visionBusy=false}
}
// started=false so the next click restarts cleanly; without it the click lands on
// togglePause and answers "Paused" for an app that is not running.
function fatal(msg){convState='idle';face('');hideVision();speakingMouth(false);say(esc(msg),true);started=false;handsFree=true}
function speakingMouth(on){if(on){smile.style.display='none';mouth.style.display='block'}else{smile.style.display='';mouth.style.display='none';setMouth(0)}}
function setMouth(open){mouth.setAttribute('ry',(2+Math.max(0,Math.min(1,open))*23).toFixed(1))}

async function startAll(){
 if(started)return;started=true;
 say('Starting up…');
 try{micStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}})}
 catch(e){started=false;fatal('Microphone blocked. Allow mic access for this page, then reload.');return}
 audioCtx=new (window.AudioContext||window.webkitAudioContext)();
 if(audioCtx.state==='suspended')await audioCtx.resume();
 const src=audioCtx.createMediaStreamSource(micStream);
 analyser=audioCtx.createAnalyser();analyser.fftSize=1024;analyser.smoothingTimeConstant=0.3;
 src.connect(analyser);dataArr=new Float32Array(analyser.fftSize);
 // Capture raw PCM instead of MediaRecorder WebM/Opus. The UNO Q's lightweight
 // audio stack reads WAV directly and does not need FFmpeg/PyAV transcoding.
 pcmNode=audioCtx.createScriptProcessor(2048,1,1);
 silentGain=audioCtx.createGain();silentGain.gain.value=0;
 src.connect(pcmNode);pcmNode.connect(silentGain);silentGain.connect(audioCtx.destination);
 pcmNode.onaudioprocess=ev=>{
   if(!started||!handsFree)return;
   const block=new Float32Array(ev.inputBuffer.getChannelData(0));
   if(convState==='listening'){
     pcmPreroll.push(block);if(pcmPreroll.length>12)pcmPreroll.shift();
   }else if(convState==='recording'){
     if(!pcmChunks.length)pcmChunks=pcmPreroll.slice();
     pcmChunks.push(block);
   }
 };
 calibrated=false;calibVals=[];noiseFloor=0.01;lastT=performance.now();
 startCam();beginListen();requestAnimationFrame(vadLoop);
}
function energy(){analyser.getFloatTimeDomainData(dataArr);let s=0;for(let i=0;i<dataArr.length;i++){const v=dataArr[i];s+=v*v}return Math.sqrt(s/dataArr.length)}

function beginListen(){
 // A queued restart must not silently undo a pause the user asked for.
 if(!handsFree||!started){convState='idle';face('');return}
 convState='listening';face('listening');say('Listening — just talk. No button to hold.');
 hasSpeech=false;voicedMs=0;silenceMs=0;speechMs=0;segStart=performance.now();pending=null;
 pcmChunks=[];pcmPreroll=[];
}
function endpoint(){if(convState!=='recording')return;convState='thinking';face('thinking');say('Thinking…');showVision('analysis');sendUtterance()}
function restartSeg(){beginListen()}

function vadLoop(){
 if(!started)return;
 const now=performance.now();let dt=now-lastT;lastT=now;if(dt>250)dt=0;else if(dt>100)dt=100;
 const e=energy();
 const visualLevel=Math.min(1,Math.max(0,(e-noiseFloor)*18));
 document.documentElement.style.setProperty('--voice-glow',(visualLevel*70).toFixed(1)+'px');
 document.documentElement.style.setProperty('--voice-scale',(visualLevel*.04).toFixed(3));
 if(!calibrated){calibVals.push(e);if(calibVals.length>=25){const s=[...calibVals].sort((a,b)=>a-b);noiseFloor=Math.max(0.005,s[Math.floor(s.length*0.25)]);calibrated=true}}
 else if(convState==='listening'&&!hasSpeech&&e<noiseFloor*2.2){noiseFloor=Math.max(0.004,noiseFloor*0.98+e*0.02)}
 const onset=Math.max(0.018,noiseFloor*3.2),keep=Math.max(0.010,noiseFloor*1.7),barge=Math.max(0.06,noiseFloor*6.5);
 if(convState==='listening'||convState==='recording'){
   if(e>onset){voicedMs+=dt;silenceMs=0;
     if(!hasSpeech&&voicedMs>=110){hasSpeech=true;convState='recording';face('recording');say('Hearing you…')}
     if(hasSpeech)speechMs+=dt;
   }else{voicedMs=Math.max(0,voicedMs-dt);
     if(hasSpeech){silenceMs+=dt;if(silenceMs>=ENDPOINT_MS&&speechMs>=MIN_SPEECH_MS)endpoint()}}
   if(!hasSpeech&&now-segStart>LISTEN_RESET_MS)restartSeg();
   if(hasSpeech&&now-segStart>MAX_UTTER_MS)endpoint();
 }else if(convState==='speaking'&&handsFree){
   if(e>barge){voicedMs+=dt;if(voicedMs>=300)bargeIn()}else voicedMs=Math.max(0,voicedMs-dt);
 }
 requestAnimationFrame(vadLoop);
}
function pcmWavBlob(chunks,sampleRate){
 let samples=0;for(const c of chunks)samples+=c.length;
 const buf=new ArrayBuffer(44+samples*2),v=new DataView(buf);
 const str=(off,s)=>{for(let i=0;i<s.length;i++)v.setUint8(off+i,s.charCodeAt(i))};
 str(0,'RIFF');v.setUint32(4,36+samples*2,true);str(8,'WAVE');str(12,'fmt ');
 v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
 v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*2,true);
 v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,samples*2,true);
 let off=44;for(const c of chunks)for(let i=0;i<c.length;i++){
   const x=Math.max(-1,Math.min(1,c[i]));v.setInt16(off,x<0?x*32768:x*32767,true);off+=2;
 }
 return new Blob([buf],{type:'audio/wav'});
}
function downsamplePcm(chunks,inputRate,targetRate=16000){
 let total=0;for(const c of chunks)total+=c.length;
 const input=new Float32Array(total);let p=0;
 for(const c of chunks){input.set(c,p);p+=c.length}
 if(inputRate<=targetRate)return {chunks:[input],sampleRate:inputRate};
 const ratio=inputRate/targetRate,out=new Float32Array(Math.floor(input.length/ratio));
 // A box-filter average prevents the harsh aliasing produced by simply dropping
 // samples. Moonshine wants 16 kHz, so doing this in-browser also cuts upload and
 // server resampling work by about 3x on normal 48 kHz USB microphones.
 for(let i=0;i<out.length;i++){
   const a=Math.floor(i*ratio),b=Math.max(a+1,Math.floor((i+1)*ratio));let sum=0;
   for(let j=a;j<b&&j<input.length;j++)sum+=input[j];
   out[i]=sum/(Math.min(b,input.length)-a);
 }
 return {chunks:[out],sampleRate:targetRate};
}
async function sendUtterance(){
 const chunks=pcmChunks;pcmChunks=[];pcmPreroll=[];
 if(!chunks.length){hideVision();beginListen();return}
 const pcm=downsamplePcm(chunks,audioCtx.sampleRate);
 const blob=pcmWavBlob(pcm.chunks,pcm.sampleRate);
 // If streaming is unavailable, retry this same in-memory WAV through the
 // proven complete-turn endpoint. No recording and no user turn is lost.
 if(window.ReadableStream&&await sendStream(blob))return;
 await sendLegacy(blob);
}
async function sendLegacy(blob){
 const fd=new FormData();fd.append('audio',blob,'turn.wav');
 let r,j;
 try{r=await fetch('/api/voice/turn',{method:'POST',body:fd})}
 catch(e){fatal('Lost the connection to the server. Is it still running?');return}
 // A non-JSON body means the server threw before our handler (HTML 500, proxy 502).
 // Reporting that as a network error would be a lie -- the server answered.
 try{j=await r.json()}
 catch(e){fatal('Server error '+r.status+' — check the terminal running the agent.');return}
 // 503 = the server is misconfigured (bad/missing key, token budget). Every turn
 // fails identically, so stop and show it rather than looping silently forever.
 if(r.status===503){fatal((j.error||'Server not configured.').split('\\n')[0]);return}
 if(j.error){hideVision();say('<b>Heard:</b> '+esc(j.transcript||'(you)')+'\\n'+esc(j.error),true);setTimeout(beginListen,2600);return}
 if(!j.transcript){hideVision();say("Didn't catch that — say it again.");setTimeout(beginListen,900);return}
 if(j.audio_b64){say('<b>You:</b> '+esc(j.transcript)+'\\n<b>Me:</b> '+esc(j.reply));play(j.audio_b64)}
 else{hideVision();say('<b>You:</b> '+esc(j.transcript)+'\\n<b>Me:</b> '+esc(j.reply||'(no reply)')+'\\n(no voice audio)');setTimeout(beginListen,1800)}
}
async function sendStream(blob){
 const fd=new FormData();fd.append('audio',blob,'turn.wav');
 const controller=new AbortController();streamAbort=controller;streamDone=false;playbackEnd=audioCtx.currentTime;
 let r,gotMeta=false,gotAudio=false,transcript='';
 try{r=await fetch('/api/voice/turn-stream',{method:'POST',body:fd,signal:controller.signal})}
 catch(e){if(streamAbort===controller)streamAbort=null;return e.name==='AbortError'}
 if(!r.ok||!r.body||!r.body.getReader){if(streamAbort===controller)streamAbort=null;return false}
 const reader=r.body.getReader(),decoder=new TextDecoder();let pendingText='';
 try{
  while(true){
   const part=await reader.read();pendingText+=decoder.decode(part.value||new Uint8Array(),{stream:!part.done});
   const lines=pendingText.split('\n');pendingText=lines.pop();
   for(const line of lines){
    if(!line.trim())continue;let ev;try{ev=JSON.parse(line)}catch(e){throw new Error('invalid stream data')}
    if(ev.type==='meta'){
      gotMeta=true;transcript=ev.transcript||'';
      if(!transcript){hideVision();say("Didn't catch that — say it again.");continue}
      say('<b>You:</b> '+esc(transcript)+'\\n<b>Me:</b> '+esc(ev.reply||''));
    }else if(ev.type==='audio'&&ev.audio_b64){
      gotAudio=true;await queueStreamAudio(ev.audio_b64);
    }else if(ev.type==='done'){
      streamDone=true;
      if(!transcript){setTimeout(beginListen,900)}
      else if(!gotAudio){hideVision();say('<b>You:</b> '+esc(transcript)+'\\n(no voice audio)',true);setTimeout(beginListen,1800)}
      else finishStreamIfReady();
    }else if(ev.type==='error'){
      if(ev.fatal){fatal((ev.error||'Server not configured.').split('\\n')[0]);return true}
      throw new Error(ev.error||'speech stream failed');
    }
   }
   if(part.done)break;
  }
  if(!gotMeta)throw new Error('empty speech stream');
  return true;
 }catch(e){
  if(e.name==='AbortError')return true;
  if(!gotMeta)return false;
  streamDone=true;stopAllAudio();fatal(e.message||'Speech stream failed.');return true;
 }finally{if(streamAbort===controller)streamAbort=null}
}
async function queueStreamAudio(b64){
 if(!handsFree)return;
 const by=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
 let buf;try{buf=await audioCtx.decodeAudioData(by.buffer)}catch(e){throw new Error('Reply audio would not decode')}
 if(!handsFree)return;
 if(convState!=='speaking'){hideVision();convState='speaking';face('speaking');speakingMouth(true);voicedMs=0}
 const src=audioCtx.createBufferSource();src.buffer=buf;
 const an=audioCtx.createAnalyser();an.fftSize=512;const md=new Float32Array(an.fftSize);
 src.connect(an);an.connect(audioCtx.destination);streamSources.add(src);
 const when=Math.max(audioCtx.currentTime+.025,playbackEnd);playbackEnd=when+buf.duration;
 src.onended=()=>{streamSources.delete(src);finishStreamIfReady()};src.start(when);
 (function ml(){if(!streamSources.has(src))return;an.getFloatTimeDomainData(md);let s=0;for(let i=0;i<md.length;i++){const v=md[i];s+=v*v}const level=Math.min(1,Math.sqrt(s/md.length)*9);setMouth(level);document.documentElement.style.setProperty('--voice-glow',(level*70).toFixed(1)+'px');document.documentElement.style.setProperty('--voice-scale',(level*.04).toFixed(3));requestAnimationFrame(ml)})();
}
function finishStreamIfReady(){if(streamDone&&streamSources.size===0){speakingMouth(false);afterSpeak()}}
async function play(b64){
 hideVision();
 if(!handsFree){convState='idle';face('');return}   // paused during 'thinking' -> stay silent
 convState='speaking';face('speaking');speakingMouth(true);voicedMs=0;
 let buf;try{const by=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));buf=await audioCtx.decodeAudioData(by.buffer)}
 catch(e){speakingMouth(false);say('Reply came back but the audio would not play.',true);setTimeout(afterSpeak,1800);return}
 const src=audioCtx.createBufferSource();src.buffer=buf;
 const an=audioCtx.createAnalyser();an.fftSize=512;const md=new Float32Array(an.fftSize);
 src.connect(an);an.connect(audioCtx.destination);curSrc=src;
 src.onended=()=>{if(curSrc===src)curSrc=null;speakingMouth(false);afterSpeak()};
 src.start();
 (function ml(){if(curSrc!==src)return;an.getFloatTimeDomainData(md);let s=0;for(let i=0;i<md.length;i++){const v=md[i];s+=v*v}const level=Math.min(1,Math.sqrt(s/md.length)*9);setMouth(level);document.documentElement.style.setProperty('--voice-glow',(level*70).toFixed(1)+'px');document.documentElement.style.setProperty('--voice-scale',(level*.04).toFixed(3));requestAnimationFrame(ml)})();
}
function afterSpeak(){if(handsFree)beginListen();else{convState='idle';face('')}}
function stopAllAudio(){
 if(curSrc){try{curSrc.stop()}catch(e){}curSrc=null}
 for(const src of [...streamSources]){try{src.stop()}catch(e){}}
 streamSources.clear();speakingMouth(false);
}
function bargeIn(){if(streamAbort){streamAbort.abort();streamAbort=null}streamDone=true;stopAllAudio()}

let statusBusy=false;
setInterval(async()=>{
 if(!started||statusBusy)return;              // in-flight guard: polls must not queue up behind a busy vision lock
 statusBusy=true;
 try{const s=await (await fetch('/api/vision/enroll_status')).json();looking=!!s.active;face(baseFace);
   if(s.active)showVision('enroll',s);else if(visionKind==='enroll'){if(convState==='thinking')showVision('analysis');else hideVision()}
   if(visionOpen)refreshVision()}
 catch(e){}finally{statusBusy=false}
},500);

let vstream=null;
let camOk=false,frameBusy=false;
async function startCam(){
 if(vstream)return;
 // Camera denial must be visible: silently swallowing it leaves every vision tool
 // answering "no one is in view", and enroll then blames the user's lighting.
 try{vstream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240}});cam.srcObject=vstream;await cam.play();camOk=true}
 catch(e){camOk=false;say('Camera blocked — I can hear you but not see you. Allow camera access and reload.',true);return}
setInterval(()=>{if(!cam.videoWidth||frameBusy||(convState==='thinking'&&!looking)||convState==='speaking')return;
  // Pause vision inference while STT/LLM/TTS owns the small board CPU. The last
  // frame stays fresh enough for the current turn. Active enrollment is the one
  // exception: it needs fresh frames to collect the requested samples.
  vcanvas.width=cam.videoWidth;vcanvas.height=cam.videoHeight;
  vcanvas.getContext('2d').drawImage(cam,0,0,vcanvas.width,vcanvas.height);
  vcanvas.toBlob(b=>{if(!b)return;frameBusy=true;const fd=new FormData();fd.append('frame',b,'f.jpg');
   fetch('/api/vision/frame',{method:'POST',body:fd}).catch(()=>{}).finally(()=>{frameBusy=false})},'image/jpeg',0.7)},500);
}

function togglePause(){handsFree=!handsFree;if(handsFree){beginListen()}else{if(streamAbort){streamAbort.abort();streamAbort=null}streamDone=true;stopAllAudio();hideVision();pcmChunks=[];pcmPreroll=[];convState='idle';face('');say('Paused — click the face to resume.')}}
faceEl.addEventListener('click',e=>{e.stopPropagation();if(!started)startAll();else togglePause()});
document.body.addEventListener('click',()=>{if(!started)startAll()});
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&started&&(convState==='listening'||convState==='recording'))restartSeg()});
</script></body></html>
"""


def create_app(agent, vision_service):
    app = Flask(__name__)

    page = PAGE.replace("__ENDPOINT_MS__", str(config.VAD_ENDPOINT_MS))

    @app.after_request
    def disable_client_cache(response):
        # This is an appliance UI under active development. A stale cached page
        # can keep sending an obsolete audio codec even after the board is fixed.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index():
        return page

    @app.post("/api/voice/turn")
    def voice_turn():
        f = request.files.get("audio")
        if not f:
            return jsonify({"error": "no audio"}), 400
        suffix = "." + (f.filename.rsplit(".", 1)[-1] if "." in f.filename else "webm")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            f.save(tmp.name)
            try:
                with agent.turn_lock:
                    return jsonify(agent.handle_audio(tmp.name))
            except SystemExit as e:
                return jsonify({"error": str(e)}), 503
            except Exception as e:
                return jsonify({"error": f"turn failed: {e}"}), 500

    @app.post("/api/voice/turn-stream")
    def voice_turn_stream():
        """NDJSON metadata followed by independently playable sentence WAVs.

        The original JSON endpoint above stays untouched as a compatibility and
        browser fallback path. Streaming changes only when playback may begin; STT,
        LLM/tool behavior, reply text, and Piper voice are shared with the old path.
        """
        f = request.files.get("audio")
        if not f:
            return jsonify({"error": "no audio"}), 400
        suffix = "." + (f.filename.rsplit(".", 1)[-1] if "." in f.filename else "wav")
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        f.save(tmp.name)

        def event(kind, **fields):
            return json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n"

        def generate():
            try:
                with agent.turn_lock:
                    try:
                        out = agent.understand_audio(tmp.name)
                    except SystemExit as e:
                        yield event("error", error=str(e), fatal=True)
                        return
                    except Exception as e:
                        yield event("error", error=f"turn failed: {e}", fatal=False)
                        return

                    yield event("meta", transcript=out.get("transcript", ""),
                                reply=out.get("reply", ""), tools=out.get("tools", []),
                                timings_ms=out.get("timings_ms", {}))
                    if not out.get("transcript"):
                        yield event("done", timings_ms=out.get("timings_ms", {}), chunks=0)
                        return

                    chunks = sentence_chunks(out.get("reply", ""))
                    t_tts = time.perf_counter()
                    first_ms = None
                    for index, text_part in enumerate(chunks):
                        wav = agent.tts.synth(text_part)
                        if first_ms is None:
                            first_ms = (time.perf_counter() - t_tts) * 1000
                        yield event("audio", index=index, text=text_part,
                                    audio_b64=base64.b64encode(wav).decode() if wav else "")
                    tts_ms = (time.perf_counter() - t_tts) * 1000
                    timings = dict(out.get("timings_ms", {}))
                    timings.update(
                        tts=round(tts_ms, 1),
                        first_audio=round(first_ms or 0.0, 1),
                        total=round(timings.get("stt", 0.0) + timings.get("llm", 0.0) + tts_ms, 1),
                    )
                    print(f"[voice] stream: stt={timings.get('stt', 0):.0f}ms "
                          f"llm={timings.get('llm', 0):.0f}ms first_audio={first_ms or 0:.0f}ms "
                          f"tts={tts_ms:.0f}ms chunks={len(chunks)}", flush=True)
                    yield event("done", timings_ms=timings, chunks=len(chunks))
            except GeneratorExit:
                return
            except Exception as e:
                yield event("error", error=f"speech stream failed: {e}", fatal=False)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

        return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

    @app.post("/api/voice/text")
    def voice_text():
        data = request.get_json(force=True)
        try:
            with agent.turn_lock:
                return jsonify(agent.handle_text(str(data.get("text", ""))))
        except SystemExit as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": f"turn failed: {e}"}), 500

    @app.get("/api/vision/scene")
    def scene():
        return jsonify(vision_service.describe_scene())

    @app.get("/api/vision/enroll_status")
    def enroll_status():
        return jsonify(vision_service.enroll_status())

    @app.post("/api/vision/frame")
    def vision_frame():
        f = request.files.get("frame")
        if f is None:
            return jsonify({"ok": False, "reason": "no frame"}), 400
        buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "reason": "decode failed"}), 400
        return jsonify(vision_service.submit_frame(img))

    return app
