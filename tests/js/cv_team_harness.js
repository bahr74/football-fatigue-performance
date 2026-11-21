// Runs the team-analytics script of cv.html against a saved /api/cv/analytics payload with a fake DOM.
// usage: node cv_team_harness.js CV_HTML PAYLOAD_JSON   -> prints the rendered innerHTML/textContent by element id
const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8'),payload=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
const which=process.argv[3+1]||'team'; // 'team' = second script; 'coach' = second + third (the coach script uses its helpers)
const code=which==='coach'?scripts[1]+'\n'+scripts[2]:scripts[1];
const noop=new Proxy(function(){},{get:(t,k)=>k==='canvas'?{}:noop,apply:()=>noop,set:()=>true});
const els={};
const mk=id=>els[id]||(els[id]={id,innerHTML:'',textContent:'',disabled:false,style:{},clientWidth:900,width:900,height:200,
  addEventListener(){},getContext:()=>noop,getBoundingClientRect:()=>({left:0,top:0,width:900,height:200})});
const document={getElementById:mk,createElement:()=>mk('tmp'+Math.random()),querySelectorAll:()=>[]};
const fetch=async u=>({ok:true,status:200,json:async()=>(String(u).includes('/coach')&&payload.coach)?payload.coach:(payload.analytics||payload)});
new Function('document','window','fetch','video','sourceFrame','frameBase','activeOffset','mode','switchSource','setInterval','clearInterval',code)(
  document,{addEventListener(){},devicePixelRatio:1},fetch,{addEventListener(){},readyState:4},()=>0,0,0,'detector',()=>{},()=>1,()=>{});
setTimeout(()=>{const out={};for(const [k,v] of Object.entries(els))out[k]=(v.innerHTML||'')+(v.textContent||'')+(v.disabled?'[disabled]':'');console.log(JSON.stringify(out));process.exit(0);},300);
