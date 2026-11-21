// Regression harness: runs the real cv.html script against a live server, simulating continuous playback and seeks.
// usage: node cv_playback_harness.js PORT CV_HTML [FRAMES] [SEEK_LIST]
const fs=require('fs'),http=require('http');
const port=process.argv[2],html=fs.readFileSync(process.argv[3],'utf8'),code=html.match(/<script>([\s\S]*?)<\/script>/)[1]; // first script = playback; the second one is team analytics
const els={};let raf=null;
const mk=id=>els[id]||(els[id]={id,textContent:'',innerHTML:'',style:{},dataset:{},classList:{toggle(){},add(){}},addEventListener(t,f){(this.h=this.h||{})[t]=f},getBoundingClientRect:()=>({width:800,height:200}),getContext:()=>new Proxy({},{get:()=>()=>{},set:()=>true}),width:600,height:150,duration:239,currentTime:0,paused:true});
const document={getElementById:mk,addEventListener(){}};const window={addEventListener(){}};
const calls=[];const errors=[];
const fetch=async u=>{calls.push(u);const r=await new Promise((res,rej)=>http.get(`http://127.0.0.1:${port}${u}`,x=>{let b='';x.on('data',c=>b+=c);x.on('end',()=>res(b));}).on('error',rej));return {ok:true,status:200,json:async()=>JSON.parse(r)};};
const rafs=[];const requestAnimationFrame=f=>{rafs.push(f);return rafs.length;};
new Function('document','window','fetch','setInterval','clearInterval','requestAnimationFrame',code)(document,window,fetch,()=>1,()=>{},requestAnimationFrame);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const v=mk('video');
const shown=()=>els.detections.textContent;
async function at(frame,wait=60){v.currentTime=frame/25;v.h.seeked&&v.h.seeked();v.h.timeupdate&&v.h.timeupdate();rafs.splice(0).forEach(f=>f());await sleep(wait);v.h.timeupdate&&v.h.timeupdate();rafs.splice(0).forEach(f=>f());await sleep(5);return shown();}
(async()=>{
 await sleep(1500);
 els['detector-mode'].h.click();
 v.paused=false;const missing=[],N=Number(process.argv[4]||340);
 for(let f=0;f<N;f++){const t=await at(f,f%100===0?60:0);if(!/^\d+$/.test(t))missing.push(f);}
 const seeks={};for(const f of (process.argv[5]||'304,177,0,399').split(',').map(Number))seeks[f]=await at(f,250);
 console.log(JSON.stringify({missing,seeks,error:mk('error').textContent||null}));
 process.exit(0);})();
