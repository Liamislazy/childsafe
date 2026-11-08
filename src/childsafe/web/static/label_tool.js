/* Capture one still photo, draw boxes, save locally; review contact sheets. No video, no uploads. */
'use strict';
const $=id=>document.getElementById(id);
const NAMES={knife:'Dao',fork:'Nĩa',scissors:'Kéo',pen:'Bút'};
const COLORS={knife:'#e5484d',fork:'#f0a92b',scissors:'#3b6fd8',pen:'#2e9e6b'};
let stream=null,shot=null,boxes=[],drag=null,lastSaved=null;

async function info(){
  const data=await (await fetch('/api/info',{cache:'no-store'})).json();
  const o=data.own_boxes;
  $('saved').textContent=`Ảnh tự chụp: ${data.counts.own.total} ảnh · dao ${o.knife||0}, nĩa ${o.fork||0}, kéo ${o.scissors||0}, bút ${o.pen||0}, ảnh nền ${data.own_background}`;
  return data;
}
function canSave(){ $('save').disabled=!(shot&&$('consent').checked&&$('complete').checked); }
function drawShot(){
  const cv=$('draw'),ctx=cv.getContext('2d');
  ctx.drawImage(shot,0,0,cv.width,cv.height);
  ctx.lineWidth=Math.max(2,cv.width/320);ctx.font=`${Math.max(14,cv.width/45)}px system-ui`;
  const all=drag?[...boxes,drag]:boxes;
  for(const b of all){ctx.strokeStyle=ctx.fillStyle=COLORS[b.label];
    ctx.strokeRect(b.x0*cv.width,b.y0*cv.height,(b.x1-b.x0)*cv.width,(b.y1-b.y0)*cv.height);
    ctx.fillText(NAMES[b.label],b.x0*cv.width+4,b.y0*cv.height+ctx.lineWidth*8);}
  const list=$('boxes');list.replaceChildren();
  boxes.forEach((b,i)=>{const li=document.createElement('li');li.textContent=NAMES[b.label]+' ';
    const x=document.createElement('button');x.textContent='Xóa khung';x.onclick=()=>{boxes.splice(i,1);drawShot();};
    li.append(x);list.append(li);});
}
function point(e){const cv=$('draw'),r=cv.getBoundingClientRect();
  // object-fit:contain letterboxes the canvas; map the pointer into the image area.
  const s=Math.min(r.width/cv.width,r.height/cv.height),w=cv.width*s,h=cv.height*s,ox=(r.width-w)/2,oy=(r.height-h)/2;
  return [Math.min(1,Math.max(0,(e.clientX-r.left-ox)/w)),Math.min(1,Math.max(0,(e.clientY-r.top-oy)/h))];}
$('draw').onpointerdown=e=>{if(!shot)return;const [x,y]=point(e);drag={label:$('label').value,sx:x,sy:y,x0:x,y0:y,x1:x,y1:y};$('draw').setPointerCapture(e.pointerId);};
$('draw').onpointermove=e=>{if(!drag)return;const [x,y]=point(e);
  Object.assign(drag,{x0:Math.min(drag.sx,x),x1:Math.max(drag.sx,x),y0:Math.min(drag.sy,y),y1:Math.max(drag.sy,y)});drawShot();};
$('draw').onpointerup=()=>{if(drag&&drag.x1-drag.x0>.01&&drag.y1-drag.y0>.01){const {label,x0,y0,x1,y1}=drag;boxes.push({label,x0,y0,x1,y1});}
  drag=null;drawShot();};
$('camera').onclick=async()=>{
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;$('camera').textContent='Bật camera';$('shoot').disabled=true;return;}
  try{stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    $('video').srcObject=stream;$('camera').textContent='Tắt camera';$('shoot').disabled=false;$('error').textContent='';}
  catch(e){$('error').textContent='Không mở được camera: '+e.message;}
};
$('shoot').onclick=()=>{
  const v=$('video'),cv=$('draw');if(!v.videoWidth)return;
  cv.width=v.videoWidth;cv.height=v.videoHeight;const c=document.createElement('canvas');c.width=cv.width;c.height=cv.height;
  c.getContext('2d').drawImage(v,0,0);shot=c;boxes=[];$('complete').checked=false;
  cv.hidden=false;v.hidden=true;$('retake').disabled=false;drawShot();canSave();
};
$('retake').onclick=()=>{shot=null;boxes=[];$('draw').hidden=true;$('video').hidden=false;$('retake').disabled=true;$('boxes').replaceChildren();canSave();};
['consent','complete'].forEach(id=>$(id).onchange=canSave);
$('save').onclick=async()=>{
  $('save').disabled=true;$('error').textContent='';
  const body={image:shot.toDataURL('image/jpeg',.92),person:$('person').value.trim(),session:$('session').value.trim(),
    boxes,consent:$('consent').checked,complete:$('complete').checked};
  const r=await fetch('/api/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const out=await r.json();
  if(!r.ok){$('error').textContent=out.error||'Không lưu được.';canSave();return;}
  lastSaved=out.saved;$('undo').disabled=false;$('retake').click();info();
};
$('undo').onclick=async()=>{
  if(!lastSaved)return;const r=await fetch('/api/capture?'+new URLSearchParams({image:lastSaved}),{method:'DELETE'});
  if(r.ok){lastSaved=null;$('undo').disabled=true;info();}
};

// ---- review -----------------------------------------------------------------------------------
let page=[];
async function loadPage(){
  $('reviewError').textContent='';
  const set=$('set').value,data=await (await fetch('/api/review?'+new URLSearchParams({set,limit:24}),{cache:'no-store'})).json();
  const all=(await info()).counts[set];
  $('progress').textContent=`Đã duyệt ${all.reviewed}/${all.total} · loại ${all.rejected} · còn ${data.total}`;
  page=data.items.map(item=>({...item,verdict:'ok'}));
  const sheet=$('sheet');sheet.replaceChildren();
  if(!page.length){sheet.textContent='Không còn ảnh chưa duyệt trong bộ này.';}
  for(const item of page){
    const card=document.createElement('div');card.className='thumb';const tag=document.createElement('span');tag.className='tag';tag.textContent='Loại';
    const cv=document.createElement('canvas');card.append(cv,tag);sheet.append(card);
    const img=new Image();img.onload=()=>{const s=Math.min(1,480/img.width);cv.width=img.width*s;cv.height=img.height*s;
      const ctx=cv.getContext('2d');ctx.drawImage(img,0,0,cv.width,cv.height);ctx.lineWidth=3;ctx.font='bold 15px system-ui';
      for(const b of item.boxes){ctx.strokeStyle=ctx.fillStyle=COLORS[b.label]||'#fff';
        ctx.strokeRect(b.x0*cv.width,b.y0*cv.height,(b.x1-b.x0)*cv.width,(b.y1-b.y0)*cv.height);
        ctx.fillText(NAMES[b.label]||b.label,b.x0*cv.width+3,Math.max(15,b.y0*cv.height-4));}};
    img.src='/files/'+item.image.split('/').map(encodeURIComponent).join('/');
    card.onclick=()=>{item.verdict=item.verdict==='ok'?'reject':'ok';card.classList.toggle('reject',item.verdict==='reject');note();};
  }
  note();
}
function note(){const bad=page.filter(i=>i.verdict==='reject').length;$('pageNote').textContent=page.length?`${page.length-bad} đúng, ${bad} loại`:'';$('savePage').disabled=!page.length;}
$('savePage').onclick=async()=>{
  const verdicts=Object.fromEntries(page.map(i=>[i.image,i.verdict]));
  const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({set:$('set').value,verdicts})});
  if(!r.ok){$('reviewError').textContent='Không lưu được trang này.';return;}
  loadPage();
};
$('set').onchange=loadPage;
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));
  $('capture').hidden=b.dataset.tab!=='capture';$('review').hidden=b.dataset.tab!=='review';
  if(b.dataset.tab==='review')loadPage();
});
info();
