"""
Signature visuals, rendered as small self-contained HTML/JS components:
  globe_html(...)      orbital Earth with precise project pins, auto tour between programs
  structure_html(...)  pixel blocks that fly in and assemble into the structure being built
                       (data center, canal, dam); the build stops at the project's % complete
"""
import json
import os

import streamlit.components.v1 as components

# Pinned public CDN builds (set MPCT_ASSET_BASE to serve the same files locally)
GLOBE_JS = os.environ.get("MPCT_GLOBE_JS", "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/dist/globe.gl.min.js")
IMG_BASE = os.environ.get("MPCT_IMG_BASE", "https://cdn.jsdelivr.net/npm/three-globe@2.45.6/example/img/")

GLOBE_TEMPLATE = r"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;background:transparent;overflow:hidden;font-family:'Figtree',system-ui,sans-serif;}
#wrap{position:relative;width:100%;height:__H__px;border-radius:24px;overflow:hidden;
  background:radial-gradient(ellipse at 55% 45%,#ffffff 0%,#e8f0fe 55%,#d3e3fd 100%);border:1px solid #e3e6ea;}
#g{position:absolute;inset:0;}
.hud{position:absolute;color:#444746;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;pointer-events:none;}
#fallback{position:absolute;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;gap:8px;color:#444746;font-size:15px;text-align:center;padding:0 40px}
#fallback b{color:#1f1f1f;font-size:18px}
.tl{top:14px;left:16px}.tr{top:14px;right:16px;text-align:right}.bl{bottom:14px;left:16px}.br{bottom:14px;right:16px;text-align:right}
.big{font-size:22px;letter-spacing:0;text-transform:none;color:#1f1f1f;font-weight:600}
.acc{color:#0b57d0}.dim{color:#5e6368}
.reticle{position:absolute;left:50%;top:50%;width:120px;height:120px;margin:-60px 0 0 -60px;border:1px solid rgba(11,87,208,.18);border-radius:50%;pointer-events:none}
.reticle:before,.reticle:after{content:"";position:absolute;background:rgba(11,87,208,.2)}
.reticle:before{left:50%;top:-14px;width:1px;height:148px}.reticle:after{top:50%;left:-14px;height:1px;width:148px}
.pinwrap{pointer-events:auto;cursor:default}
.pin{position:relative;transform:translate(-50%,-100%);animation:drop .9s cubic-bezier(.2,1.4,.4,1) both}
.pin .head{width:12px;height:12px;border-radius:50%;margin:0 auto;background:currentColor;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.35)}
.pin .stem{width:2px;height:22px;margin:0 auto;background:linear-gradient(currentColor,transparent)}
.pin .lab{position:absolute;left:14px;top:-6px;white-space:nowrap;font-size:11px;font-weight:600;color:#1f1f1f;
  background:#ffffff;border:1px solid #c4c7c5;padding:2px 7px;border-radius:6px;opacity:0;transition:opacity .4s;box-shadow:0 1px 3px rgba(60,64,67,.2)}
.pin.on .lab{opacity:1}
@keyframes drop{0%{transform:translate(-50%,-260%);opacity:0}100%{transform:translate(-50%,-100%);opacity:1}}
#scan{display:none}
</style></head><body>
<div id="wrap"><div id="g"></div><div id="scan"></div><div class="reticle"></div>
<div id="fallback"><b>The 3D globe could not load</b><span>It needs a network connection to its map library. Every project and its status is listed below.</span></div>
<div class="hud tl"><div class="acc">● Orbital view</div><div class="dim" id="alt">ALT —</div></div>
<div class="hud tr"><div class="dim">Active sites</div><div class="big" id="count">—</div></div>
<div class="hud bl"><div class="dim" id="prog">—</div><div class="big" id="site">Acquiring target…</div><div id="coord" class="acc">—</div></div>
<div class="hud br"><div class="dim">Status</div><div class="big" id="stat">—</div></div>
</div>
<script src="__GLOBE_JS__" onerror="window.__globeFail=true"></script>
<script>
if (typeof Globe === 'undefined' || window.__globeFail) {
  document.getElementById('fallback').style.display = 'flex';
  document.querySelectorAll('.reticle,.hud').forEach(e => e.style.display = 'none');
  throw new Error('globe library unavailable');
}
const PTS = __POINTS__;
const TOUR = __TOUR__;
const IMG = "__IMG__";
const H = __H__;
const el = document.getElementById('g');
const W0 = () => el.clientWidth || document.documentElement.clientWidth || window.innerWidth || 900;
const col = s => s==='Red' ? '#d93025' : s==='Amber' ? '#e37400' : s==='Green' ? '#188038' : '#0b57d0';
const world = Globe({animateIn:true})(el)
  .width(W0()).height(H)
  .backgroundColor('rgba(0,0,0,0)')
  .globeImageUrl(IMG+'earth-blue-marble.jpg')
  .bumpImageUrl(IMG+'earth-topology.png')
  .showAtmosphere(true).atmosphereColor('#8ab4f8').atmosphereAltitude(0.18)
  .ringsData(PTS).ringLat('lat').ringLng('lng').ringColor(d => t => `rgba(11,87,208,${(1-t)*0.8})`)
  .ringMaxRadius(1.6).ringPropagationSpeed(1.2).ringRepeatPeriod(1600)
  .htmlElementsData(PTS).htmlLat('lat').htmlLng('lng').htmlAltitude(0.005)
  .htmlElement(d => { const w=document.createElement('div'); w.className='pinwrap';
      const p=document.createElement('div'); p.className='pin'; p.id='pin-'+d.id; p.style.color=col(d.status);
      p.style.animationDelay=(1.2+d.i*0.15)+'s';
      p.innerHTML=`<div class="lab">${d.name} · ${d.lat.toFixed(4)}°, ${d.lng.toFixed(4)}°</div><div class="head"></div><div class="stem"></div>`;
      w.title = `${d.name}\n${d.location}\n${d.lat.toFixed(4)}, ${d.lng.toFixed(4)}`; w.appendChild(p); return w; });
const ctr = world.controls(); ctr.autoRotate = true; ctr.autoRotateSpeed = 0.35; ctr.enableZoom = true;
world.pointOfView({lat: 18, lng: -20, altitude: 3.4});
document.getElementById('count').textContent = PTS.length;
function altTxt(){ const a = world.pointOfView().altitude; document.getElementById('alt').textContent = 'ALT ' + Math.round(a*6371).toLocaleString() + ' km'; }
setInterval(altTxt, 200);
let k = 0;
function visit(){
  if(!TOUR.length) return;
  const t = TOUR[k % TOUR.length]; k++;
  ctr.autoRotate = false;
  world.pointOfView({lat: t.lat, lng: t.lng, altitude: t.alt}, 2600);
  document.querySelectorAll('.pin').forEach(p => p.classList.remove('on'));
  (t.ids||[]).forEach(id => { const p = document.getElementById('pin-'+id); if(p) p.classList.add('on'); });
  document.getElementById('prog').textContent = t.program;
  document.getElementById('site').textContent = t.title;
  document.getElementById('coord').textContent = t.coord;
  const s = document.getElementById('stat'); s.textContent = t.status; s.style.color = t.color;
  if(t.rotate) setTimeout(() => { ctr.autoRotate = true; ctr.autoRotateSpeed = 0.15; }, 3000);
}
setTimeout(visit, 2200); setInterval(visit, 6500);
const probe = new Image(); probe.onerror = () => { document.getElementById('fallback').style.display = 'flex';
  document.querySelectorAll('.reticle,.hud').forEach(e => e.style.display = 'none'); };
probe.src = IMG + 'earth-blue-marble.jpg';
new ResizeObserver(() => world.width(W0())).observe(document.body);
window.addEventListener('resize', () => world.width(W0()));
</script></body></html>
"""

STRUCT_TEMPLATE = r"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent;overflow:hidden}
#w{position:relative;height:__H__px;border-radius:12px;overflow:hidden;
   background:linear-gradient(180deg,#e8f0fe 0%,#f8fafd 100%);border:1px solid #e3e6ea}
canvas{display:block}
.tag{position:absolute;top:8px;left:12px;font:700 10px system-ui,sans-serif;letter-spacing:.12em;color:#0b57d0;text-transform:uppercase}
.pc{position:absolute;top:8px;right:12px;font:700 11px system-ui,sans-serif;color:#1f1f1f}
</style></head><body><div id="w"><canvas id="c"></canvas><div class="tag">__LABEL__</div><div class="pc" id="pc"></div></div>
<script>
const TYPE="__TYPE__", PROG=__PROG__, H=__H__, CELL=__CELL__, SEED=__SEED__;
const cv=document.getElementById('c'), wrap=document.getElementById('w');
const dpr=window.devicePixelRatio||1; let W=wrap.clientWidth||document.documentElement.clientWidth||window.innerWidth||600;
cv.width=W*dpr; cv.height=H*dpr; cv.style.width=W+'px'; cv.style.height=H+'px';
const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr);
function rng(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
const R=rng(SEED);
const cols=Math.floor(W/CELL), rows=Math.floor(H/CELL);
const cells=[]; // x,y,c,k(kind: g ground, s structure, l light, w water)
const put=(x,y,c,k)=>{x=Math.round(x);y=Math.round(y);if(x>=0&&x<cols&&y>=0&&y<rows)cells.push({x,y,c,k:k||'s'});};
const pick=a=>a[Math.floor(R()*a.length)];
function ground(g, greens){ for(let y=g;y<rows;y++)for(let x=0;x<cols;x++)put(x,y,pick(greens),'g'); }
if(TYPE==='datacenter'){
  const g=rows-3; ground(g,['#1a2b22','#1e3127','#16261d']);
  const bx=Math.floor(cols*.08), bw=Math.floor(cols*.56), bh=Math.floor(rows*.50), by=g-bh;
  for(let x=bx;x<bx+bw;x++)for(let y=by;y<g;y++){
    let c=((x-bx)%7===0)?'#1f2c45':'#2a3b5a'; if(y===by)c='#6f8fc2'; if(y===by+1)c='#3d5683';
    const band=y>=by+Math.floor(bh*.32)&&y<by+Math.floor(bh*.32)+3;
    if(band&&(x-bx)%7!==0){ put(x,y,R()<.5?'#4fd1ff':'#20c9a0','l'); } else put(x,y,c);
  }
  for(let x=bx+2;x<bx+bw-3;x+=5){ for(let dx=0;dx<3;dx++)for(let dy=1;dy<=2;dy++)put(x+dx,by-dy,dy===2?'#9aa6b8':'#7b889c'); }
  const b2x=bx+bw+3, b2w=Math.floor(cols*.17), b2h=Math.floor(rows*.34);
  for(let x=b2x;x<b2x+b2w;x++)for(let y=g-b2h;y<g;y++)put(x,y,y===g-b2h?'#6f8fc2':((x-b2x)%5===0?'#1f2c45':'#26364f'));
  for(let y=g-b2h+3;y<g-2;y+=3)for(let x=b2x+2;x<b2x+b2w-2;x+=2)put(x,y,'#4fd1ff','l');
  const sx=b2x+b2w+3;
  for(let t=0;t<3;t++){const tx=sx+t*4; if(tx+2>=cols)break; for(let dx=0;dx<3;dx++)for(let dy=1;dy<=4;dy++)put(tx+dx,g-dy,dy===4?'#e0b040':'#c98500');}
  for(let x=0;x<cols;x+=2)put(x,g-1,'#3a4a66');
} else if(TYPE==='canal'){
  const top=Math.floor(rows*.30);
  for(let y=top;y<rows;y++)for(let x=0;x<cols;x++){const band=Math.floor((x+y*0.6)/6)%3; put(x,y,['#1d4a2a','#245d33','#2f6e35'][band],'g');}
  const cx=cols/2, tw=cols*.62, bw=cols*.22, y0=Math.floor(rows*.40), y1=rows-4, fsl=y0+Math.floor((y1-y0)*.28);
  for(let y=y0;y<=y1;y++){ const f=(y-y0)/(y1-y0), half=(tw/2)*(1-f)+(bw/2)*f;
    for(let x=Math.floor(cx-half);x<=Math.ceil(cx+half);x++){
      const edge=Math.abs(x-(cx-half))<2.2||Math.abs(x-(cx+half))<2.2||y>=y1-1;
      if(edge)put(x,y,y>=y1-1?'#a9aaa0':(R()<.5?'#c4c5ba':'#b3b4a9'));
      else if(y>=fsl)put(x,y,(y+x)%5===0?'#5aa2f0':'#3987e5','w');
    }}
  for(let x=Math.floor(cx-tw/2)-6;x<Math.floor(cx-tw/2);x++)for(let y=y0-2;y<y0;y++)put(x,y,'#7a5a3a');
  for(let x=Math.ceil(cx+tw/2)+1;x<Math.ceil(cx+tw/2)+7;x++)for(let y=y0-2;y<y0;y++)put(x,y,'#6b6b63');
  const dk=y0-6; for(let x=Math.floor(cx-tw/2)-2;x<=Math.ceil(cx+tw/2)+2;x++){put(x,dk,'#aab3be');put(x,dk+1,'#8b95a3');}
  for(let p=-2;p<=2;p++){const px=Math.round(cx+p*(bw*.55)); for(let y=dk+2;y<=y1-1;y++){put(px,y,'#8b95a3');put(px+1,y,'#7a8492');}}
  for(let p=-2;p<2;p++){const gx=Math.round(cx+p*(bw*.55))+2, gw=Math.round(bw*.55)-2; for(let x=gx;x<gx+gw;x++)for(let y=dk+3;y<fsl-1;y++)if((y-dk)%2===0)put(x,y,'#d0a54a');}
  for(let t=0;t<5;t++){const tx=Math.floor(R()*cols*.18)+(t%2?Math.floor(cols*.80):1), ty=top+2+Math.floor(R()*4);
    for(let dx=-2;dx<=2;dx++)for(let dy=-2;dy<=1;dy++)if(dx*dx+dy*dy<6)put(tx+dx,ty+dy,R()<.5?'#2f7a3a':'#3a8a45'); put(tx,ty+2,'#5a4030');}
} else { // dam
  const g=rows-3; ground(g,['#3a3328','#332d24']);
  const dx0=Math.floor(cols*.52), crest=Math.floor(rows*.22);
  for(let y=crest;y<g;y++){ const f=(y-crest)/(g-crest), w=Math.floor(3+f*cols*.30);
    for(let x=dx0;x<dx0+w;x++)put(x,y,(x+y)%9===0?'#9aa0a8':'#b9bec5'); }
  for(let y=crest+3;y<g;y++)for(let x=0;x<dx0;x++)put(x,y,(x*3+y)%7===0?'#5aa2f0':'#2f7bd6','w');
  for(let x=dx0-1;x<dx0+5;x++)for(let y=crest-3;y<crest;y++)put(x,y,'#d0a54a');
}
const struct=cells.filter(c=>c.k!=='g'), grd=cells.filter(c=>c.k==='g');
struct.sort((a,b)=>(b.y-a.y)||(R()-.5));
const nBuilt=Math.round(struct.length*Math.max(0,Math.min(1,PROG)));
struct.forEach((c,i)=>{c.built=i<nBuilt;c.rank=i/Math.max(1,nBuilt);c.sx=c.x+(R()-.5)*cols*.9;c.sy=-R()*rows*1.2-4;c.ph=R()*6.28;});
grd.forEach(c=>{c.ph=R()*6.28;});
document.getElementById('pc').textContent=Math.round(PROG*100)+'% BUILT';
const T=4.4, ASM=2.5, HOLD=3.9; const t0=performance.now();
const ease=t=>1-Math.pow(1-t,3);
function frame(now){
  const t=((now-t0)/1000)%T; ctx.clearRect(0,0,W,H);
  const fade=t>HOLD?1-(t-HOLD)/(T-HOLD):1;
  ctx.globalAlpha=0.55; for(const c of grd){ctx.fillStyle=c.c;ctx.fillRect(c.x*CELL,c.y*CELL,CELL-1,CELL-1);} ctx.globalAlpha=1;
  for(const c of struct){
    if(!c.built){ctx.globalAlpha=0.10;ctx.fillStyle=c.c;ctx.fillRect(c.x*CELL,c.y*CELL,CELL-1,CELL-1);ctx.globalAlpha=1;continue;}
    const st=c.rank*(ASM-0.7), p=Math.max(0,Math.min(1,(t-st)/0.7)); if(p<=0)continue;
    const e=ease(p), x=c.sx+(c.x-c.sx)*e, y=c.sy+(c.y-c.sy)*e;
    let col=c.c, a=fade*(0.35+0.65*e);
    if(p>=1&&c.k==='l'){a*=0.55+0.45*Math.sin(now/260+c.ph);}
    if(p>=1&&c.k==='w'){a*=0.8+0.2*Math.sin(now/400+c.x*0.5+c.ph);}
    ctx.globalAlpha=Math.max(0,a); ctx.fillStyle=col; ctx.fillRect(x*CELL,y*CELL,CELL-1,CELL-1);
    if(p<1){ctx.globalAlpha=0.25*fade;ctx.fillStyle='#4fd1ff';ctx.fillRect(x*CELL,y*CELL-CELL*2,1,CELL*2);}
  }
  ctx.globalAlpha=1;
  const sy=((now/12)%(H+40))-20; const gr=ctx.createLinearGradient(0,sy-20,0,sy); gr.addColorStop(0,'rgba(11,87,208,0)'); gr.addColorStop(1,'rgba(11,87,208,0.06)');
  ctx.fillStyle=gr; ctx.fillRect(0,sy-20,W,20);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>
"""

STRUCT_LABEL = {"datacenter": "Data center", "canal": "Canal network", "dam": "Dam"}


def globe_html(points, tour, height=560):
    return (GLOBE_TEMPLATE.replace("__POINTS__", json.dumps(points)).replace("__TOUR__", json.dumps(tour))
            .replace("__IMG__", IMG_BASE).replace("__GLOBE_JS__", GLOBE_JS).replace("__H__", str(height)))


def structure_html(kind, progress, height=130, cell=4, seed=7, label=None):
    return (STRUCT_TEMPLATE.replace("__TYPE__", kind if kind in STRUCT_LABEL else "datacenter")
            .replace("__PROG__", f"{max(0.0, min(1.0, progress)):.3f}").replace("__H__", str(height))
            .replace("__CELL__", str(cell)).replace("__SEED__", str(int(seed)))
            .replace("__LABEL__", label or STRUCT_LABEL.get(kind, "Structure")))


def globe(points, tour, height=560):
    components.html(globe_html(points, tour, height), height=height + 6)


def structure(kind, progress, height=130, cell=4, seed=7, label=None):
    components.html(structure_html(kind, progress, height, cell, seed, label), height=height + 4)
