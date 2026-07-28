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
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context

import face_emotion as fe

from . import config
from .tts import sentence_chunks


class Busy(Exception):
    """Another turn is still running. Half-duplex is intentional, but waiting
    forever behind a wedged turn is not: the caller gets a real answer instead."""


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aware</title>
<style>
:root{
 --canvas:#08080c;
 --skin:#f3c94f;--skin-hi:#ffe786;--ink:#17140b;

 --text:#fff;--text-2:rgba(255,255,255,.60);--text-3:rgba(255,255,255,.45);--text-4:rgba(255,255,255,.34);

 --glass:rgba(255,255,255,.05);--glass-2:rgba(255,255,255,.08);
 --glass-border:rgba(255,255,255,.08);--glass-border-2:rgba(255,255,255,.13);
 --glass-shadow:0 8px 32px rgba(0,0,0,.5);

 --calm:#8ab4f8;--warm:#f5c26b;--danger:#f08b84;

 --r-sm:12px;--r-md:16px;--r-lg:20px;--r-xl:24px;--r-2xl:28px;--r-full:9999px;
 --blur-glass:16px;--blur-hero:80px;
 --ease:cubic-bezier(.4,0,.2,1);

 --voice-level:0;--voice-scale:0;--accent:var(--calm);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--canvas);color:var(--text);
 display:flex;flex-direction:column;align-items:center;justify-content:center;gap:clamp(28px,5vh,52px);
 min-height:100%;cursor:pointer;overflow:hidden;
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display",Inter,system-ui,sans-serif;
 -webkit-font-smoothing:antialiased}

/* One soft ambient wash. It shifts hue with state instead of adding chrome. */
body:before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
 background:radial-gradient(circle at 50% 40%,color-mix(in srgb,var(--accent) 12%,transparent) 0,transparent 62%);
 filter:blur(var(--blur-hero));opacity:.55;transition:opacity 1.2s var(--ease),background 1.2s var(--ease)}

/* ---------- face ---------- */
.presence{position:relative;z-index:1;display:grid;place-items:center;
 width:min(46vw,46vh);height:min(46vw,46vh);
 transition:transform .8s var(--ease),opacity .6s var(--ease),filter .6s var(--ease)}
/* A single halo that breathes with the voice level. No rings, no orbits, no spinners.
   Opacity and transform are the only per-frame properties: both are composited, so
   the voice drives them at 60fps for free. Blur is deliberately CONSTANT -- animating
   a blur radius re-rasterizes the whole gradient every frame and visibly stutters.
   Neither gets a transition either: they are already updated every frame, and a
   transition would restart 60 times a second and lag behind the voice. */
.presence:before{content:"";position:absolute;inset:-14%;border-radius:50%;pointer-events:none;
 background:radial-gradient(circle,color-mix(in srgb,var(--accent) 22%,transparent) 0,transparent 68%);
 opacity:calc(var(--halo-base) + var(--voice-level) * .34);
 transform:scale(calc(1 + var(--voice-scale)));
 filter:blur(30px);will-change:opacity,transform;
 transition:background .9s var(--ease)}
body{--halo-base:.14}
body[data-state="listening"]{--halo-base:.30}
body[data-state="recording"]{--halo-base:.42}
body[data-state="thinking"]{--halo-base:.28}
body[data-state="speaking"]{--halo-base:.38}

body[data-state="listening"]{--accent:var(--calm)}
body[data-state="recording"]{--accent:var(--calm)}
body[data-state="thinking"]{--accent:#b6a7f0}
body[data-state="speaking"]{--accent:var(--warm)}

#face{position:relative;z-index:2;width:100%;height:100%;overflow:visible;
 filter:drop-shadow(0 18px 40px rgba(0,0,0,.45));
 transition:opacity .6s var(--ease),filter .6s var(--ease),transform .8s var(--ease)}
#face.off{opacity:.62;filter:saturate(.55) drop-shadow(0 14px 32px rgba(0,0,0,.35))}
.skin{fill:url(#skinGradient)}
.eye{fill:var(--ink);transform-box:fill-box;transform-origin:center;animation:blink 6.5s var(--ease) infinite}
@keyframes blink{0%,96%,100%{transform:scaleY(1)}98%{transform:scaleY(.1)}}
#smile{fill:none;stroke:var(--ink);stroke-width:9;stroke-linecap:round}
#mouth{fill:var(--ink)}
/* Thinking: one slow, shallow breath. Calm, not a loading spinner. */
body[data-state="thinking"] #face{animation:breathe 3.4s var(--ease) infinite}
@keyframes breathe{0%,100%{transform:scale(.99)}50%{transform:scale(1.01)}}
body[data-vision="true"] .presence{transform:scale(.9);opacity:.12;filter:blur(6px)}

/* ---------- readout ---------- */
.readout{position:relative;z-index:1;display:flex;flex-direction:column;align-items:center;gap:10px;
 width:min(86vw,560px);text-align:center;
 transition:opacity .6s var(--ease),transform .6s var(--ease)}
body[data-vision="true"] .readout{opacity:.30;transform:translateY(8px)}
#mode{font-size:13px;font-weight:500;color:var(--text-4);transition:color .5s var(--ease)}
body[data-state="listening"] #mode,body[data-state="recording"] #mode,
body[data-state="thinking"] #mode,body[data-state="speaking"] #mode{color:var(--text-3)}
#status{font-size:clamp(15px,1.9vh,17px);line-height:1.6;font-weight:400;color:var(--text-2);
 min-height:3.2em;white-space:pre-wrap;cursor:default;transition:color .4s var(--ease)}
#status b{color:var(--text);font-weight:550}
#status.err{color:var(--danger)}

/* ---------- camera ---------- */
.vision-lens{position:fixed;z-index:8;left:50%;top:50%;width:min(82vw,520px);
 transform:translate(-50%,-50%) scale(.97);opacity:0;visibility:hidden;pointer-events:none;
 background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--r-2xl);
 backdrop-filter:blur(var(--blur-glass)) saturate(1.3);-webkit-backdrop-filter:blur(var(--blur-glass)) saturate(1.3);
 box-shadow:var(--glass-shadow);padding:14px;
 transition:opacity .5s var(--ease),transform .6s var(--ease),visibility 0s linear .6s}
body[data-vision="true"] .vision-lens{opacity:1;visibility:visible;transform:translate(-50%,-50%) scale(1);transition-delay:0s}
.vision-frame{position:relative;aspect-ratio:4/3;overflow:hidden;border-radius:var(--r-lg);background:#101014;
 border:1px solid var(--glass-border)}
#cam{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1);
 filter:saturate(.85) brightness(.86)}
/* The robot's own eye, streamed from the board. Deliberately NOT mirrored: this
   is a third-person view of what the robot sees, not a selfie preview. */
#boardCam{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
 filter:saturate(.85) brightness(.86)}
body[data-eye="board"] #cam{display:none}
body[data-eye="browser"] #boardCam{display:none}
#visionMarks{position:absolute;inset:0}
.track{position:absolute;border:1.5px solid rgba(255,255,255,.55);border-radius:var(--r-sm);
 min-width:34px;min-height:34px;box-shadow:0 2px 12px rgba(0,0,0,.28);
 transition:left .45s var(--ease),top .45s var(--ease),width .45s var(--ease),height .45s var(--ease)}
.track-label{position:absolute;left:0;bottom:-32px;display:inline-flex;align-items:baseline;gap:7px;
 white-space:nowrap;padding:4px 10px;border-radius:var(--r-full);
 background:rgba(10,10,14,.62);border:1px solid var(--glass-border-2);
 backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
 color:var(--text);font-size:12px;font-weight:500}
.track-label em{font-style:normal;color:var(--text-3);font-weight:400}
.vision-empty{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
 color:var(--text-4);font-size:13px;font-weight:400;opacity:0;transition:opacity .4s var(--ease)}
.vision-empty.show{opacity:1}
.vision-foot{display:flex;align-items:baseline;gap:10px;padding:12px 4px 2px}
#visionTitle{color:var(--text);font-size:14px;font-weight:500;letter-spacing:-.01em}
#visionSummary{color:var(--text-4);font-size:13px;font-weight:400;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.enroll-link{position:fixed;z-index:9;right:clamp(16px,2.5vw,28px);bottom:clamp(16px,2.5vh,26px);
 padding:9px 15px;border-radius:var(--r-full);color:var(--text-3);font-size:13px;font-weight:500;
 text-decoration:none;background:var(--glass);border:1px solid var(--glass-border);
 backdrop-filter:blur(var(--blur-glass));-webkit-backdrop-filter:blur(var(--blur-glass));
 transition:color .3s var(--ease),background .3s var(--ease),border-color .3s var(--ease)}
.enroll-link:hover{color:var(--text);background:var(--glass-2);border-color:var(--glass-border-2)}
body[data-vision="true"] .enroll-link{opacity:0;pointer-events:none}
.link{position:fixed;top:clamp(16px,2.6vh,26px);right:clamp(16px,2.6vw,26px);z-index:9;
 display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border-radius:var(--r-full);
 background:var(--glass);border:1px solid var(--glass-border);
 backdrop-filter:blur(var(--blur-glass)) saturate(1.3);-webkit-backdrop-filter:blur(var(--blur-glass)) saturate(1.3);
 color:var(--text-3);font-size:12px;font-weight:500;text-decoration:none;cursor:pointer;
 transition:color .35s var(--ease),border-color .35s var(--ease),background .35s var(--ease)}
.link:hover{color:var(--text);border-color:var(--glass-border-2);background:var(--glass-2)}
/* Board connection. A steady dot, never a pulsing one: it reports state, it does not nag. */
#board{left:clamp(16px,2.6vw,26px);right:auto}
#boardDot{width:7px;height:7px;border-radius:50%;background:var(--text-4);
 transition:background .4s var(--ease)}
body[data-board="connected"] #boardDot{background:#6fcf97}
body[data-board="connecting"] #boardDot{background:var(--warm)}
body[data-board="offline"] #boardDot{background:var(--danger)}
body[data-board="offline"] #board{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 35%,transparent)}
@media (max-height:620px){
 .presence{width:min(38vw,38vh);height:min(38vw,38vh)}
 .vision-lens{width:min(64vw,420px)}
 #status{min-height:2.6em}
}
@media (prefers-reduced-motion:reduce){
 *,*:before,*:after{animation-duration:.001ms!important;animation-iteration-count:1!important;
  transition-duration:.001ms!important}
}
</style></head>
<body data-state="idle" data-looking="false" data-vision="false" data-eye="browser">
 <div class="presence" aria-hidden="true">
  <svg id="face" class="face off" viewBox="0 0 200 200">
   <defs><radialGradient id="skinGradient" cx="38%" cy="28%" r="78%">
    <stop offset="0" stop-color="var(--skin-hi)"/><stop offset=".58" stop-color="var(--skin)"/>
    <stop offset="1" stop-color="#dcae2e"/></radialGradient></defs>
   <circle class="skin" cx="100" cy="100" r="94"/>
   <circle class="eye" cx="70" cy="84" r="10"/>
   <circle class="eye" cx="130" cy="84" r="10"/>
   <path id="smile" d="M62 122 Q100 156 138 122"/>
   <ellipse id="mouth" cx="100" cy="130" rx="27" ry="3" style="display:none"/>
  </svg>
 </div>
 <button id="board" class="link" type="button" title="Check the board connection"><span id="boardDot"></span><span id="boardText">Connecting</span></button>
 <div class="readout" aria-live="polite">
  <div id="mode"><span id="modeText">Standby</span></div>
  <div id="status">Click the face to begin</div>
 </div>
 <section id="visionLens" class="vision-lens" aria-hidden="true">
  <div class="vision-frame">
   <video id="cam" autoplay playsinline muted></video>
   <img id="boardCam" alt="">
   <div id="visionMarks"></div>
   <div id="visionEmpty" class="vision-empty">No face in view</div>
  </div>
  <div class="vision-foot"><span id="visionTitle">Looking</span><span id="visionSummary"></span></div>
 </section>
 <a class="enroll-link" id="enrollLink" target="_blank" rel="noopener">Manage people</a>
 <canvas id="vcanvas" style="display:none"></canvas>
<script>
const faceEl=document.getElementById('face'),smile=document.getElementById('smile'),mouth=document.getElementById('mouth'),
 statusEl=document.getElementById('status'),modeText=document.getElementById('modeText'),
 cam=document.getElementById('cam'),vcanvas=document.getElementById('vcanvas'),
 boardCam=document.getElementById('boardCam'),
 visionLens=document.getElementById('visionLens'),visionTitle=document.getElementById('visionTitle'),
 visionMarks=document.getElementById('visionMarks'),visionEmpty=document.getElementById('visionEmpty'),
 visionSummary=document.getElementById('visionSummary');
let started=false,handsFree=true,looking=false,baseFace='';
let micStream=null,audioCtx=null,analyser=null,dataArr=null,lastT=0;
let calibrated=false,calibVals=[],noiseFloor=0.01;
let convState='idle',pcmNode=null,silentGain=null,pcmChunks=[],pcmPreroll=[],segStart=0,curSrc=null;
let streamSources=new Set(),streamDone=false,playbackEnd=0,streamAbort=null;
let voicedMs=0,silenceMs=0,speechMs=0,hasSpeech=false,pending=null,peakE=0;
let visionOpen=false,visionKind='',visionBusy=false,turnUsedCamera=false;
// 'browser' = this page supplies the camera (laptop dev). 'board' = the robot has
// its own webcam and owns perception; we must not open a second camera or upload
// competing frames, so the lens views the robot's eye instead. Learned from
// /api/health before the camera is ever requested.
let frameSource='browser';
let outLevel=0,speakStart=0,frameTick=0;   // live playback level + when speaking began, for barge-in
const API=__API_BASE__;
// When this page is hosted, its own /enroll route serves the enrollment UI.
// When the board serves it directly, that route does not exist, so go to the
// enrollment service on its own port.
const ENROLL_URL=API?'/enroll':location.protocol+'//'+location.hostname+':8000';   // '' when the board serves the page; absolute when Vercel does
const ENDPOINT_MS=__ENDPOINT_MS__,MIN_SPEECH_MS=340,MAX_UTTER_MS=15000,LISTEN_RESET_MS=8000;
// Someone talking across the room clears the onset threshold but never gets
// close to the level of the person addressing the device. Requiring a near-field
// peak drops that audio before it is ever uploaded, so it costs no latency either.
const NEAR_FIELD_MIN=0.055,NEAR_FIELD_MULT=6;
// Barge-in must survive the speaker-to-mic path. Without a grace window, a hold
// long enough to exclude a syllable of echo, and a threshold that rises with the
// current playback level, the reply interrupts ITSELF and is never heard.
const BARGE_GRACE_MS=900,BARGE_HOLD_MS=650,BARGE_BLEED=0.85;

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
 }else{visionTitle.textContent='Looking';visionSummary.textContent='Checking who is there'}
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
   visionTitle.textContent='Got the frame';visionSummary.textContent='Thinking about it';return;
 }
 visionMarks.replaceChildren();visionEmpty.classList.toggle('show',!people.length);
 for(const p of people){
   const b=p.bbox||[0,0,0,0],mark=document.createElement('div'),label=document.createElement('div');
   // The browser preview is mirrored (it is a selfie view), so the detector's x
   // must be mirrored to match. The robot's own eye is not mirrored, so it must
   // not be -- getting this wrong puts every box on the wrong side of the frame.
   const left=frameSource==='board'?(b[0]*100/fw):(100-(b[0]+b[2])*100/fw);
   mark.className='track';mark.style.left=left+'%';mark.style.top=(b[1]*100/fh)+'%';
   mark.style.width=(b[2]*100/fw)+'%';mark.style.height=(b[3]*100/fh)+'%';label.className='track-label';
   const name=p.name&&p.name!=='unknown'?p.name:'Unrecognized';
   const emotion=p.emotion&&p.sentiment!=='not_enabled'?p.emotion:'';
   label.innerHTML='<span>'+esc(name)+'</span><em>'+(emotion?esc(emotion)+', '+pct(p.emotion_score):pct(p.identity_score))+'</em>';
   mark.appendChild(label);visionMarks.appendChild(mark);
 }
 if(visionKind!=='enroll'){
   visionTitle.textContent=people.length?'Recognizing '+people.length+(people.length===1?' person':' people'):'Checking the scene';
   visionSummary.textContent=people.length?people.map(p=>(p.name&&p.name!=='unknown'?p.name:'Unknown')+(p.emotion?', '+p.emotion:'')).join('   '):'No face currently detected';
 }
}
async function refreshVision(){
 if(!visionOpen||visionBusy)return;visionBusy=true;
 try{const r=await fetch(API+'/api/vision/scene');if(r.ok)renderVision(await r.json())}catch(e){}finally{visionBusy=false}
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
 // A browser with no Web Audio (or one that refuses to resume the context) would
 // otherwise leave the page "started" with a dead VAD and no way back.
 try{
  audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  if(audioCtx.state==='suspended')await audioCtx.resume();
 }catch(e){fatal('This browser blocked audio playback. Try Chrome or Safari, then reload.');return}
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
     pcmPreroll.push(block);if(pcmPreroll.length>6)pcmPreroll.shift();
   }else if(convState==='recording'){
     if(!pcmChunks.length)pcmChunks=pcmPreroll.slice();
     pcmChunks.push(block);
   }
 };
 calibrated=false;calibVals=[];noiseFloor=0.01;lastT=performance.now();
 // Settle the camera question BEFORE asking for hardware: on a standalone robot
 // the board owns the camera, and prompting for the laptop's webcam there is both
 // a pointless permission dialog and a second feed the board would have to reject.
 await checkBoard();
 if(frameSource==='board'){camOk=true;startBoardEye()}else{startCam()}
 beginListen();requestAnimationFrame(vadLoop);
}
// Poll the robot's own camera into the lens. Only while the lens is open: every
// frame costs the board a JPEG encode, and it is closed most of the time.
let boardEyeTimer=null;
function startBoardEye(){
 if(boardEyeTimer)return;
 boardEyeTimer=setInterval(()=>{
  if(!started||!visionOpen)return;
  boardCam.src=API+'/api/vision/snapshot.jpg?t='+Date.now();
 },400);
}
let voiceSmooth=0;
function setVoiceLevel(level){
 // The analyser is jumpy frame to frame; unsmoothed it reads as flicker rather
 // than breath. One exponential filter here serves both the mic and playback.
 voiceSmooth+=(Math.min(1,Math.max(0,level||0))-voiceSmooth)*.28;
 const st=document.documentElement.style;
 st.setProperty('--voice-level',voiceSmooth.toFixed(3));
 st.setProperty('--voice-scale',(voiceSmooth*.05).toFixed(4));
}
function energy(){analyser.getFloatTimeDomainData(dataArr);let s=0;for(let i=0;i<dataArr.length;i++){const v=dataArr[i];s+=v*v}return Math.sqrt(s/dataArr.length)}

function beginListen(){
 // A queued restart must not silently undo a pause the user asked for.
 if(!handsFree||!started){convState='idle';face('');return}
 hideVision();   // a new turn starts clean; last turn's camera panel closes here
 convState='listening';face('listening');say('Listening');
 hasSpeech=false;voicedMs=0;silenceMs=0;speechMs=0;segStart=performance.now();pending=null;peakE=0;
 pcmChunks=[];pcmPreroll=[];
}
// The camera panel is evidence of looking, not decoration: it opens only after
// the model actually calls a tool that reads the camera. Opening it on every turn
// put a face-scanning overlay on "what time is it".
const CAMERA_TOOLS=new Set(['who_is_in_view','describe_scene','get_person_emotion','enroll_face','train_emotion','start_watching']);
function usedCamera(trace){return Array.isArray(trace)&&trace.some(t=>CAMERA_TOOLS.has(t&&t.tool))}
function endpoint(){if(convState!=='recording')return;
 if(peakE<Math.max(NEAR_FIELD_MIN,noiseFloor*NEAR_FIELD_MULT)){beginListen();return}
 turnUsedCamera=false;convState='thinking';face('thinking');say('Thinking…');sendUtterance()}
function restartSeg(){beginListen()}

function vadLoop(){
 if(!started)return;
 const now=performance.now();let dt=now-lastT;lastT=now;if(dt>250)dt=0;else if(dt>100)dt=100;
 const e=energy();
 const visualLevel=Math.min(1,Math.max(0,(e-noiseFloor)*18));
 setVoiceLevel(visualLevel);
 if(!calibrated){calibVals.push(e);if(calibVals.length>=25){const s=[...calibVals].sort((a,b)=>a-b);noiseFloor=Math.max(0.005,s[Math.floor(s.length*0.25)]);calibrated=true}}
 else if(convState==='listening'&&!hasSpeech&&e<noiseFloor*2.2){noiseFloor=Math.max(0.004,noiseFloor*0.98+e*0.02)}
 const onset=Math.max(0.032,noiseFloor*4.2),keep=Math.max(0.010,noiseFloor*1.7),barge=Math.max(0.06,noiseFloor*6.5);
 if(convState==='listening'||convState==='recording'){
   if(e>peakE)peakE=e;
   if(e>onset){voicedMs+=dt;silenceMs=0;
     if(!hasSpeech&&voicedMs>=110){hasSpeech=true;convState='recording';face('recording');say('Hearing you…')}
     if(hasSpeech)speechMs+=dt;
   }else{voicedMs=Math.max(0,voicedMs-dt);
     if(hasSpeech){silenceMs+=dt;if(silenceMs>=ENDPOINT_MS&&speechMs>=MIN_SPEECH_MS)endpoint()}}
   if(!hasSpeech&&now-segStart>LISTEN_RESET_MS)restartSeg();
   if(hasSpeech&&now-segStart>MAX_UTTER_MS)endpoint();
 }else if(convState==='speaking'&&handsFree){
   // Speaker bleed rises and falls with the reply, so the bar to interrupt rises
   // with it. Decay faster than it accumulates so echo can never creep to the hold.
   const thresh=Math.max(barge,noiseFloor*9,outLevel*BARGE_BLEED);
   if(now-speakStart>BARGE_GRACE_MS&&e>thresh){voicedMs+=dt;if(voicedMs>=BARGE_HOLD_MS)bargeIn()}
   else voicedMs=Math.max(0,voicedMs-dt*1.5);
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
 let chunks=pcmChunks;pcmChunks=[];pcmPreroll=[];
 // ENDPOINT_MS of silence is what proved the turn ended; only a short tail is
 // needed to avoid clipping the last consonant. Transcribing the rest is waste.
 const blockMs=2048/audioCtx.sampleRate*1000,drop=Math.floor((ENDPOINT_MS-120)/blockMs);
 if(drop>0&&chunks.length>drop+4)chunks=chunks.slice(0,chunks.length-drop);
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
 try{r=await fetch(API+'/api/voice/turn',{method:'POST',body:fd})}
 catch(e){fatal('Lost the connection to the server. Is it still running?');return}
 // A non-JSON body means the server threw before our handler (HTML 500, proxy 502).
 // Reporting that as a network error would be a lie -- the server answered.
 try{j=await r.json()}
 catch(e){fatal('Server error '+r.status+'. Check the terminal running the agent.');return}
 // 503 = the server is misconfigured (bad/missing key, token budget). Every turn
 // fails identically, so stop and show it rather than looping silently forever.
 if(r.status===503){fatal((j.error||'Server not configured.').split('\\n')[0]);return}
 if(j.error){hideVision();say('<b>Heard:</b> '+esc(j.transcript||'(you)')+'\\n'+esc(j.error),true);setTimeout(beginListen,2600);return}
 if(!j.transcript){hideVision();say("Didn't catch that. Say it again.");setTimeout(beginListen,900);return}
 if(j.audio_b64){say('<b>You:</b> '+esc(j.transcript)+'\\n<b>Me:</b> '+esc(j.reply));play(j.audio_b64)}
 else{hideVision();say('<b>You:</b> '+esc(j.transcript)+'\\n<b>Me:</b> '+esc(j.reply||'(no reply)')+'\\n(no voice audio)');setTimeout(beginListen,1800)}
}
async function sendStream(blob){
 const fd=new FormData();fd.append('audio',blob,'turn.wav');
 const controller=new AbortController();streamAbort=controller;streamDone=false;playbackEnd=audioCtx.currentTime;
 let r,gotMeta=false,gotAudio=false,transcript='';
 try{r=await fetch(API+'/api/voice/turn-stream',{method:'POST',body:fd,signal:controller.signal})}
 catch(e){if(streamAbort===controller)streamAbort=null;return e.name==='AbortError'}
 if(!r.ok||!r.body||!r.body.getReader){if(streamAbort===controller)streamAbort=null;return false}
 const reader=r.body.getReader(),decoder=new TextDecoder();let pendingText='';
 try{
  while(true){
   const part=await reader.read();pendingText+=decoder.decode(part.value||new Uint8Array(),{stream:!part.done});
   const lines=pendingText.split('\\n');pendingText=lines.pop();
   for(const line of lines){
    if(!line.trim())continue;let ev;try{ev=JSON.parse(line)}catch(e){throw new Error('invalid stream data')}
    if(ev.type==='meta'){
      gotMeta=true;transcript=ev.transcript||'';
      if(usedCamera(ev.tools)){turnUsedCamera=true;showVision('analysis')}
      if(!transcript){hideVision();say("Didn't catch that. Say it again.");continue}
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
 // On a camera turn the panel stays up through the reply, so you can see what it
// saw while it tells you. Otherwise it was never opened and this is a no-op.
 if(convState!=='speaking'){if(!turnUsedCamera)hideVision();convState='speaking';face('speaking');speakingMouth(true);voicedMs=0;speakStart=performance.now()}
 const src=audioCtx.createBufferSource();src.buffer=buf;
 const an=audioCtx.createAnalyser();an.fftSize=512;const md=new Float32Array(an.fftSize);
 src.connect(an);an.connect(audioCtx.destination);streamSources.add(src);
 const when=Math.max(audioCtx.currentTime+.025,playbackEnd);playbackEnd=when+buf.duration;
 src.onended=()=>{streamSources.delete(src);finishStreamIfReady()};src.start(when);
 (function ml(){if(!streamSources.has(src))return;an.getFloatTimeDomainData(md);let s=0;for(let i=0;i<md.length;i++){const v=md[i];s+=v*v}const level=Math.min(1,Math.sqrt(s/md.length)*9);outLevel=level;setMouth(level);setVoiceLevel(level);requestAnimationFrame(ml)})();
}
function finishStreamIfReady(){if(streamDone&&streamSources.size===0){speakingMouth(false);afterSpeak()}}
async function play(b64){
 hideVision();
 if(!handsFree){convState='idle';face('');return}   // paused during 'thinking' -> stay silent
 convState='speaking';face('speaking');speakingMouth(true);voicedMs=0;speakStart=performance.now();
 let buf;try{const by=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));buf=await audioCtx.decodeAudioData(by.buffer)}
 catch(e){speakingMouth(false);say('Reply came back but the audio would not play.',true);setTimeout(afterSpeak,1800);return}
 const src=audioCtx.createBufferSource();src.buffer=buf;
 const an=audioCtx.createAnalyser();an.fftSize=512;const md=new Float32Array(an.fftSize);
 src.connect(an);an.connect(audioCtx.destination);curSrc=src;
 src.onended=()=>{if(curSrc===src)curSrc=null;speakingMouth(false);afterSpeak()};
 src.start();
 (function ml(){if(curSrc!==src)return;an.getFloatTimeDomainData(md);let s=0;for(let i=0;i<md.length;i++){const v=md[i];s+=v*v}const level=Math.min(1,Math.sqrt(s/md.length)*9);outLevel=level;setMouth(level);setVoiceLevel(level);requestAnimationFrame(ml)})();
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
 try{const s=await (await fetch(API+'/api/vision/enroll_status')).json();looking=!!s.active;face(baseFace);
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
 catch(e){camOk=false;say('Camera blocked. I can hear you but not see you. Allow camera access, then reload.',true);return}
setInterval(()=>{
  // Paused means paused: a user who clicked the face to stop the agent must not
  // keep streaming their camera to the server. Enrollment is the one exception --
  // it is already an explicit, in-progress request for the camera.
  if(!started||(!handsFree&&!looking))return;
  // Pause vision inference while STT/LLM/TTS owns the small board CPU. The last
  // frame stays fresh enough for the current turn. Active enrollment is the one
  // exception: it needs fresh frames to collect the requested samples.
  if(!cam.videoWidth||frameBusy||(convState==='thinking'&&!looking)||convState==='speaking')return;
  // The camera panel is closed most of the time now, and every uploaded frame
  // costs the board a face-detection pass. Halve the rate when nothing is
  // watching: identity stays fresh enough, and the cores go to STT and TTS.
  if(!visionOpen&&!looking&&(frameTick++&1))return;
  // Claim the slot before toBlob, not inside its callback: encoding is async, so
  // the next tick would otherwise fire a second upload while this one encodes.
  frameBusy=true;
  vcanvas.width=cam.videoWidth;vcanvas.height=cam.videoHeight;
  vcanvas.getContext('2d').drawImage(cam,0,0,vcanvas.width,vcanvas.height);
  vcanvas.toBlob(b=>{if(!b){frameBusy=false;return}const fd=new FormData();fd.append('frame',b,'f.jpg');
   fetch(API+'/api/vision/frame',{method:'POST',body:fd}).catch(()=>{}).finally(()=>{frameBusy=false})},'image/jpeg',0.7)},500);
}

function togglePause(){handsFree=!handsFree;if(handsFree){beginListen()}else{if(streamAbort){streamAbort.abort();streamAbort=null}streamDone=true;stopAllAudio();hideVision();pcmChunks=[];pcmPreroll=[];convState='idle';face('');say('Paused. Click the face to resume.')}}
faceEl.addEventListener('click',e=>{e.stopPropagation();if(!started)startAll();else togglePause()});
document.body.addEventListener('click',()=>{if(!started)startAll()});
document.getElementById('enrollLink').href=ENROLL_URL;
// ---------- board connection ----------
// The page is served from Vercel but the agent lives on the Arduino behind
// `adb forward`. Unplugging the cable kills the tunnel silently, so poll a cheap
// health route and say plainly which of the three states we are in.
const boardEl=document.getElementById('board'),boardDot=document.getElementById('boardDot'),
      boardText=document.getElementById('boardText');
let boardState='',boardChecking=false,boardFails=0;
function setBoard(state,label){
 if(boardState===state&&!label)return;
 boardState=state;document.body.dataset.board=state;
 boardText.textContent=label||({connected:'Board connected',connecting:'Connecting',offline:'Board disconnected'}[state]||state);
}
async function checkBoard(manual){
 if(boardChecking)return boardState==='connected';
 boardChecking=true;
 if(manual)setBoard('connecting','Searching');
 try{
  const ctl=new AbortController(),t=setTimeout(()=>ctl.abort(),2500);
  const r=await fetch(API+'/api/health',{signal:ctl.signal,cache:'no-store'});
  clearTimeout(t);
  if(!r.ok)throw new Error('bad status');
  // One page serves both deployments; the board tells us which one this is.
  try{const h=await r.json();if(h&&h.frame_source){frameSource=h.frame_source;
   document.body.dataset.eye=frameSource==='board'?'board':'browser'}}catch(e){}
  boardFails=0;setBoard('connected');return true;
 }catch(e){
  // One missed probe during a busy turn is normal; only call it offline once a
  // couple in a row fail, so the indicator does not flicker mid-conversation.
  boardFails++;
  if(boardFails>=2)setBoard('offline');else if(boardState!=='connected')setBoard('connecting');
  return false;
 }finally{boardChecking=false}
}
boardEl.addEventListener('click',e=>{e.stopPropagation();checkBoard(true)});
checkBoard();
setInterval(()=>checkBoard(),3000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&started&&(convState==='listening'||convState==='recording'))restartSeg()});
</script></body></html>
"""


def create_app(agent, vision_service):
    app = Flask(__name__)
    # A 15 s utterance at 16 kHz mono is under 500 KB and a JPEG frame is ~30 KB.
    # Cap the body so a runaway or hostile upload cannot exhaust the board's RAM;
    # Flask answers an over-size request with 413 before reading it.
    app.config["MAX_CONTENT_LENGTH"] = int(config.MAX_UPLOAD_BYTES)

    page = (PAGE.replace("__ENDPOINT_MS__", str(config.VAD_ENDPOINT_MS))
                .replace("__API_BASE__", '""'))   # same-origin when the board serves it

    # A copy of the UI can be hosted elsewhere (Vercel) and still drive this board
    # over the laptop's forwarded port. Browsers treat http://127.0.0.1 as a
    # trustworthy origin, so an HTTPS page may call it, but the request is still
    # cross-origin AND public->private: it needs CORS plus the Private Network
    # Access opt-in, including on the preflight.
    @app.after_request
    def allow_hosted_ui(response):
        origin = request.headers.get("Origin")
        if fe.is_allowed_ui_origin(origin, extra=config.ALLOWED_UI_ORIGINS):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @app.route("/api/<path:_rest>", methods=["OPTIONS"])
    def preflight(_rest):
        return ("", 204)

    @contextmanager
    def turn_slot():
        """Serialize turns, but never queue behind one forever.

        The agent is deliberately half-duplex, so a second request must wait. It
        must not wait *unboundedly*: if a turn wedges (a stalled TTS subprocess, a
        hung HTTP read), every later request used to block with no reply at all --
        the UI sat in "Thinking…" until reload, and the board audio loop went deaf.
        A bounded wait turns that into an answer the caller can recover from."""
        if not agent.turn_lock.acquire(timeout=config.TURN_LOCK_TIMEOUT):
            raise Busy("I'm still finishing the last answer — give me a second.")
        try:
            # Slow perception for the duration of the turn. The browser already
            # holds back its own frame uploads while thinking and speaking; this
            # applies the same policy to a board that owns its camera, where
            # nothing else would.
            with vision_service.turn_in_progress():
                yield
        finally:
            agent.turn_lock.release()

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

    @app.get("/favicon.ico")
    def favicon():
        # Browsers ask for this on every load; answer it instead of logging a 404.
        return Response(b"", mimetype="image/x-icon")

    def _suffix(filestorage, default):
        name = filestorage.filename or ""
        ext = name.rsplit(".", 1)[-1] if "." in name else default
        # A filename arrives from the browser: keep only a plain extension so it
        # can never steer the temp path.
        return "." + ("".join(c for c in ext if c.isalnum())[:8] or default)

    @app.post("/api/voice/turn")
    def voice_turn():
        f = request.files.get("audio")
        if not f:
            return jsonify({"error": "no audio"}), 400
        with tempfile.NamedTemporaryFile(suffix=_suffix(f, "webm"), delete=True) as tmp:
            f.save(tmp.name)
            try:
                with turn_slot():
                    return jsonify(agent.handle_audio(tmp.name))
            except Busy as e:
                return jsonify({"error": str(e)}), 409
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
        tmp = tempfile.NamedTemporaryFile(suffix=_suffix(f, "wav"), delete=False)
        tmp.close()
        f.save(tmp.name)

        def event(kind, **fields):
            return json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n"

        def generate():
            # Wall clock for the WHOLE turn, started before the lock is taken so
            # queueing counts too. This is the number the person waiting actually
            # experiences; every per-stage timing below is a component of it.
            t_turn = time.perf_counter()
            try:
                with turn_slot():
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
                    first_chunk_ms = None
                    first_audio_ms = None
                    for index, text_part in enumerate(chunks):
                        wav = agent.tts.synth(text_part)
                        if first_chunk_ms is None:
                            # Synthesis cost of the opening chunk alone (tune with
                            # VOICE_LEAD_CHUNK_MAX) ...
                            first_chunk_ms = (time.perf_counter() - t_tts) * 1000
                            # ... versus the silence the person actually sits
                            # through: STT + thinking + that first chunk. This used
                            # to report only the former under the name first_audio,
                            # which made the turn look several times more
                            # responsive than it was.
                            first_audio_ms = (time.perf_counter() - t_turn) * 1000
                        yield event("audio", index=index, text=text_part,
                                    audio_b64=base64.b64encode(wav).decode() if wav else "")
                    tts_ms = (time.perf_counter() - t_tts) * 1000
                    timings = dict(out.get("timings_ms", {}))
                    timings.update(
                        tts=round(tts_ms, 1),
                        tts_first_chunk=round(first_chunk_ms or 0.0, 1),
                        first_audio=round(first_audio_ms or 0.0, 1),
                        total=round((time.perf_counter() - t_turn) * 1000, 1),
                    )
                    print(f"[voice] stream: stt={timings.get('stt', 0):.0f}ms "
                          f"llm={timings.get('llm', 0):.0f}ms "
                          f"first_audio={first_audio_ms or 0:.0f}ms "
                          f"(chunk1 synth {first_chunk_ms or 0:.0f}ms) "
                          f"tts={tts_ms:.0f}ms chunks={len(chunks)}", flush=True)
                    yield event("done", timings_ms=timings, chunks=len(chunks))
            except GeneratorExit:
                return
            except Busy as e:
                # Not fatal: the client falls back to the plain endpoint and can retry.
                yield event("error", error=str(e), fatal=False)
            except Exception as e:
                yield event("error", error=f"speech stream failed: {e}", fatal=False)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

        return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

    @app.post("/api/voice/text")
    def voice_text():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "expected a JSON object with a 'text' field"}), 400
        try:
            with turn_slot():
                return jsonify(agent.handle_text(str(data.get("text", ""))))
        except Busy as e:
            return jsonify({"error": str(e)}), 409
        except SystemExit as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": f"turn failed: {e}"}), 500

    @app.get("/api/vision/scene")
    def scene():
        return jsonify(vision_service.describe_scene())

    @app.get("/api/health")
    def health():
        """Cheap liveness probe for the hosted UI's connection indicator.

        Deliberately touches nothing expensive: the page polls this every couple of
        seconds to tell 'board unplugged' apart from 'board busy with a turn'.

        `frame_source` is what lets one page serve both deployments. On a
        standalone robot the board owns the camera, so the browser must NOT open
        its own and must not upload frames -- it views the robot's eye instead.
        """
        return jsonify({"ok": True, "stt": config.STT_BACKEND, "tts": config.TTS_BACKEND,
                        "model": config.CEREBRAS_MODEL,
                        "frame_source": vision_service.frame_source(),
                        "watching": vision_service.running})

    @app.get("/api/vision/snapshot.jpg")
    def snapshot():
        """What the robot's own camera is looking at, for a remote viewer."""
        jpeg = vision_service.snapshot_jpeg()
        if jpeg is None:
            return ("", 503)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/vision/enroll_status")
    def enroll_status():
        return jsonify(vision_service.enroll_status())

    @app.post("/api/vision/frame")
    def vision_frame():
        f = request.files.get("frame")
        if f is None:
            return jsonify({"ok": False, "reason": "no frame"}), 400
        raw = f.read()
        # cv2.imdecode ASSERTS on an empty buffer instead of returning None, and a
        # truncated upload (tab closing mid-POST) delivers exactly that -- which
        # used to surface as an HTML 500 and a stack trace per dropped frame.
        if not raw:
            return jsonify({"ok": False, "reason": "empty frame"}), 400
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "reason": "decode failed"}), 400
        try:
            return jsonify(vision_service.submit_frame(img))
        except Exception as e:
            # One bad frame must never take down the perception endpoint.
            return jsonify({"ok": False, "reason": f"frame rejected: {e}"}), 400

    return app
