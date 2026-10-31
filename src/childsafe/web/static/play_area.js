/* No external scripts, images, analytics or persistence. Camera only after a click. */
'use strict';
class PlayAreaHistory {
  constructor(){this.items=[];this.next=1;this.fresh=false;}
  static iou(a,b){const w=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]));const h=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));const x=w*h;return x/Math.max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x);}
  update(result){
    this.fresh=!!result.usable_for_comparison;
    // Poor images may suggest a candidate for manual review, but never clear
    // an older item. Failed inference cannot publish either kind of evidence.
    if(!this.fresh&&(!result.detector_ok||!result.candidates.length))return;
    const cs=result.candidates, old=this.items.filter(t=>t.misses<2);
    const possible=cs.map(c=>old.filter(t=>c.label===t.label&&PlayAreaHistory.iou(c.box,t.box)>.2));
    const matched=new Set();
    cs.forEach((c,i)=>{
      const one=possible[i];let item;
      if(one.length===1&&possible.filter(p=>p.includes(one[0])).length===1)item=one[0];
      else {if(this.items.length>=80)throw Error('Phiên đã đủ 80 mục. Tải số liệu rồi chọn Phiên mới.');item={id:this.next++,handled:false,falsePositive:false,misses:0};this.items.push(item);}
      Object.assign(item,c,{misses:0,everSeen:true});matched.add(item.id);
    });
    if(this.fresh)this.items.forEach(t=>{if(!matched.has(t.id))t.misses=Math.min(2,(t.misses||0)+1);});
  }
  invalidate(){this.fresh=false;}
}
if(typeof module!=='undefined')module.exports={PlayAreaHistory};
if(typeof document!=='undefined'){
const $=id=>document.getElementById(id), video=$('video'), photo=$('photo');
let history=new PlayAreaHistory(),stream=null,generation=0,busy=false,pending=false,timer=null,lastGood=0,lastResult=null,sourceKind='none',photoUrl=null,currentPhoto=null,activeRequest=null;
let sessionStarted=new Date().toISOString(),reports=[],info=null,lastVideoTime=-1;
const work=document.createElement('canvas');
const canvas=$('overlay');
function status(message,review=false){$('status').textContent=message;$('status').classList.toggle('review',review);}
function blankOverlay(){canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);}
function render(){
  const el=$('items');el.replaceChildren();
  $('visibleCount').textContent=history.fresh?history.items.filter(t=>t.misses===0).length:'—';
  $('handledCount').textContent=history.items.filter(t=>t.handled).length;
  if(!history.items.length){const p=document.createElement('p');p.className='empty';p.textContent='Chưa có mục được ghi nhận; vẫn cần kiểm tra thủ công.';el.append(p);}
  history.items.forEach(item=>{
    const card=document.createElement('article');card.className='item';card.dataset.item=String(item.id);
    const h=document.createElement('h3');h.textContent=`#${item.id} · Vật giống ${item.name.toLowerCase()}`;card.append(h);
    const state=document.createElement('p');state.className='state';
    state.textContent=!history.fresh?'Chưa có ảnh đủ rõ để cập nhật mục này':item.misses===0?'Máy vẫn đang thấy trong ảnh':item.misses===1?'Chưa thấy lại — đang chờ lần kiểm tra tiếp':'Không còn phát hiện qua hai ảnh đủ rõ; bạn cần tự xác nhận';card.append(state);
    const action=document.createElement('p');action.className='note';action.textContent=item.action;card.append(action);
    if(item.at_edge){const p=document.createElement('p');p.className='note';p.textContent='Vật sát viền; hãy kiểm tra phần ngoài khung.';card.append(p);}
    const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=item.handled;
    check.onchange=()=>{item.handled=check.checked;item.confirmedAt=new Date().toISOString();render();};label.append(check,document.createTextNode(' Tôi xác nhận đã xử lý'));card.append(label);
    if(item.handled&&item.misses===0&&history.fresh){const p=document.createElement('p');p.className='note';p.textContent='Bạn đã xác nhận, nhưng máy vẫn phát hiện vật. Hãy đối chiếu trực tiếp.';card.append(p);}
    const wrong=document.createElement('button');wrong.textContent=item.falsePositive?'Bỏ ghi nhận nhận diện nhầm':'Ghi nhận nhận diện nhầm';wrong.onclick=()=>{item.falsePositive=!item.falsePositive;render();};card.append(wrong);
    el.append(card);
  });
}
function draw(result){
  const b=canvas.getBoundingClientRect();canvas.width=Math.round(b.width*devicePixelRatio);canvas.height=Math.round(b.height*devicePixelRatio);
  const ctx=canvas.getContext('2d'),s=Math.min(canvas.width/result.width,canvas.height/result.height),ox=(canvas.width-result.width*s)/2,oy=(canvas.height-result.height*s)/2;
  ctx.lineWidth=2*devicePixelRatio;ctx.strokeStyle='#ffffff';ctx.setLineDash([7*devicePixelRatio,5*devicePixelRatio]);
  const [x,y,x2,y2]=result.roi;ctx.strokeRect(ox+x*s,oy+y*s,(x2-x)*s,(y2-y)*s);ctx.setLineDash([]);
  result.candidates.forEach(c=>{const [a,t,b,z]=c.box;ctx.strokeStyle='#f0a92b';ctx.fillStyle='#f0a92b';ctx.font=`${14*devicePixelRatio}px system-ui`;ctx.strokeRect(ox+a*s,oy+t*s,(b-a)*s,(z-t)*s);ctx.fillText(`Vật giống ${c.name}`,ox+a*s,Math.max(16*devicePixelRatio,oy+t*s-5));});
}
function invalidate(message){history.invalidate();blankOverlay();render();if(message)status(message);}
function resetHistory(){history=new PlayAreaHistory();lastResult=null;lastGood=0;reports=[];sessionStarted=new Date().toISOString();document.querySelectorAll('[data-manual]').forEach(x=>x.checked=false);invalidate('Phiên mới. Giữ camera cố định và kiểm tra lại vùng bàn.');}
function stop(message='Camera đã tắt; danh sách cũ chỉ là lịch sử.'){
  generation++;pending=false;clearTimeout(timer);timer=null;
  if(activeRequest)activeRequest.abort();activeRequest=null;busy=false;
  if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;
  $('camera').textContent='Bật camera';$('capture').disabled=true;lastVideoTime=-1;
  $('source').textContent='Camera đã tắt · không ghi hình';
  invalidate(message);
}
async function start(){
  if(stream||pending){stop();return;}
  pending=true;const g=++generation;$('camera').textContent='Hủy bật camera';
  try{
    const id=$('device').value;
    const acquired=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},...(id?{deviceId:{exact:id}}:{facingMode:'environment'})},audio:false});
    if(g!==generation){acquired.getTracks().forEach(t=>t.stop());return;}
    stream=acquired;sourceKind='camera';currentPhoto=null;video.srcObject=stream;video.hidden=false;photo.hidden=true;$('placeholder').hidden=true;
    await video.play();if(g!==generation)return;
    pending=false;resetHistory();$('camera').textContent='Tắt camera';$('capture').disabled=false;
    $('source').textContent='Camera đang bật · chỉ phần trong viền được rà soát · không ghi hình';
    stream.getTracks().forEach(t=>t.addEventListener('ended',()=>{if(stream===acquired)stop('Nguồn camera bị ngắt. Bật lại để bắt đầu phiên mới.');}));
    const devices=await navigator.mediaDevices.enumerateDevices();if(g!==generation)return;
    const selected=id;$('device').replaceChildren(new Option('Mặc định',''));
    devices.filter(d=>d.kind==='videoinput').forEach((d,i)=>$('device').add(new Option(d.label||`Camera ${i+1}`,d.deviceId)));
    $('device').value=selected;schedule(g);
  }catch(e){if(g===generation){stop();$('error').textContent='Không mở được camera: '+e.message;}}
}
function schedule(g){clearTimeout(timer);if(stream&&g===generation&&$('auto').checked)timer=setTimeout(async()=>{await capture(g);schedule(g);},800);}
async function send(blob,g,kind,captured){
  if(busy||g!==generation)return;busy=true;$('error').textContent='';
  const controller=new AbortController();activeRequest=controller;const timeout=setTimeout(()=>controller.abort(),15000);
  try{
    const response=await fetch('/api/play-area/scan?'+new URLSearchParams({inset:Number($('inset').value)/100}),{method:'POST',body:blob,headers:{'X-ChildSafe-Scan':'1'},signal:controller.signal});
    const result=await response.json();if(g!==generation)return;
    if(!response.ok)throw Error(result.error||'Không phân tích được ảnh');
    if(kind==='camera'&&performance.now()-captured>3000)throw Error('Kết quả camera quá cũ; hãy chụp lại.');
    lastResult=result;draw(result);history.update(result);lastGood=performance.now();render();
    const messages=[result.message,...result.quality.reasons,!result.detector_ok?'Detector chưa hoạt động; không dùng ảnh này để kết luận vật biến mất.':''].filter(Boolean);
    status(messages.join(' '),result.candidates.length>0);
    $('timing').textContent=`${kind==='photo'?'Ảnh đã chọn, không phải cảnh hiện tại':'Ảnh camera vừa chụp'} · ${result.width}×${result.height} · xử lý ${result.ms} ms · ngưỡng chất lượng chưa hiệu chuẩn`;
    reports.push({at:new Date().toISOString(),source:kind,result});if(reports.length>20)reports.shift();
  }catch(e){if(g===generation){$('error').textContent=e.message;invalidate('Chưa có kết quả mới. Giữ nguyên các mục cần kiểm tra trước đó.');}}
  finally{clearTimeout(timeout);if(activeRequest===controller){activeRequest=null;busy=false;}}
}
async function capture(g=generation){
  if(!stream||busy||g!==generation||!video.videoWidth||!video.videoHeight)return;
  if(video.currentTime===lastVideoTime)return;lastVideoTime=video.currentTime;
  work.width=video.videoWidth;work.height=video.videoHeight;const captured=performance.now();work.getContext('2d').drawImage(video,0,0);
  const blob=await new Promise(r=>work.toBlob(r,'image/jpeg',.9));if(blob&&g===generation)await send(blob,g,'camera',captured);
}
$('camera').onclick=start;$('capture').onclick=()=>sourceKind==='photo'&&currentPhoto?send(currentPhoto,generation,'photo',performance.now()):capture();
$('auto').onchange=()=>{if(stream)schedule(generation);};
$('reset').onclick=()=>{generation++;pending=false;if(activeRequest)activeRequest.abort();activeRequest=null;busy=false;clearTimeout(timer);resetHistory();if(stream)schedule(generation);else $('camera').textContent='Bật camera';};
$('device').onchange=()=>{if(stream){stop();start();}else resetHistory();};
$('inset').oninput=()=>{$('insetValue').textContent=$('inset').value+'%';generation++;pending=false;if(activeRequest)activeRequest.abort();activeRequest=null;busy=false;resetHistory();if(stream)schedule(generation);};
$('upload').onchange=async()=>{
  const f=$('upload').files[0];if(!f)return;stop();const g=generation;sourceKind='photo';currentPhoto=f;resetHistory();
  if(f.size>8*1024*1024){$('error').textContent='Ảnh vượt 8 MB.';return;}
  if(photoUrl)URL.revokeObjectURL(photoUrl);photoUrl=URL.createObjectURL(f);
  photo.src=photoUrl;
  try{await photo.decode();if(g!==generation)return;photo.hidden=false;video.hidden=true;$('placeholder').hidden=true;$('capture').disabled=false;$('source').textContent='Ảnh do bạn chọn · không phải cảnh đang diễn ra';await send(f,g,'photo',performance.now());}catch(e){$('error').textContent='Không đọc được ảnh: '+e.message;}
  $('upload').value='';
};
$('export').onclick=()=>{
  const manual=Object.fromEntries([...document.querySelectorAll('[data-manual]')].map(x=>[x.dataset.manual,x.checked]));
  const report={schema:1,scope:'Fixed table candidates only; not a safety certificate',started:sessionStarted,exported:new Date().toISOString(),image_data_included:false,source:sourceKind,model:info,items:history.items,manual_checks_self_reported:manual,recent_scans:reports};
  const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='childsafe-play-area.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(stream||pending))stop('Tab đã ẩn; camera dừng. Bật lại để kiểm tra tiếp.');});
addEventListener('pagehide',()=>stop());addEventListener('resize',()=>{if(lastResult&&history.fresh)draw(lastResult);});
setInterval(()=>{if(stream&&lastGood&&performance.now()-lastGood>3500&&history.fresh)invalidate('Ảnh cập nhật đã cũ; chưa dùng để kết luận vật còn hay mất.');},500);
fetch('/api/play-area/info').then(r=>{if(!r.ok)throw Error('Không đọc được model');return r.json();}).then(data=>{info=data;$('scope').textContent=data.scope;$('model').textContent=`Detector: ${data.model} · các lớp đang có: ${data.supported.map(x=>x.name).join(', ')||'chưa có'} · ngưỡng chưa được kiểm chứng`;if(!data.detector_ok)status('Chưa nạp được detector cục bộ. Kiểm tra trọng số trước khi thử.');}).catch(e=>{$('error').textContent=e.message;});
}
