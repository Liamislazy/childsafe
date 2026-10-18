// Optional real-browser regression: run against the local demo with synthetic video only.
// CHILDSAFE_PLAYWRIGHT_MODULE may point to an installed Playwright module.
const {chromium}=require(process.env.CHILDSAFE_PLAYWRIGHT_MODULE || 'playwright');
const url=process.env.CHILDSAFE_TEST_URL || 'http://127.0.0.1:8765';
(async()=>{
  let ready=false;
  for(let attempt=0;attempt<60;attempt++){
    try {if((await fetch(url+'/api/info')).ok){ready=true;break}} catch(_){}
    await new Promise(resolve=>setTimeout(resolve,500));
  }
  if(!ready) throw new Error('local demo did not become ready');
  const browser=await chromium.launch({channel:process.env.CHILDSAFE_BROWSER_CHANNEL || 'msedge',headless:true});
  try {
    const page=await browser.newPage(); const errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.addInitScript(()=>{
      navigator.mediaDevices.getUserMedia=async()=>{
        const canvas=document.createElement('canvas'); canvas.width=320;canvas.height=240;
        const ctx=canvas.getContext('2d');let n=0;
        const timer=setInterval(()=>{ctx.fillStyle='#777';ctx.fillRect(0,0,320,240);ctx.fillStyle='#aaa';ctx.fillRect(n++%300,30,10,10)},40);
        const stream=canvas.captureStream(25);
        stream.getVideoTracks()[0].addEventListener('ended',()=>clearInterval(timer));
        return stream;
      };
    });
    await page.goto(url);
    await page.waitForFunction(()=>document.getElementById('runtimeStatus').textContent.includes('Sẵn sàng'));
    await page.getByRole('button',{name:'Bật camera',exact:true}).click();
    await page.waitForFunction(()=>running);
    let release; const blocked=new Promise(resolve=>{release=resolve});
    let entered; const renewalEntered=new Promise(resolve=>{entered=resolve});
    let first=true;
    await page.route('**/api/camera/renew?*',async route=>{
      if(!first) return route.continue(); first=false;
      const response=await route.fetch(); entered(); await blocked; await route.fulfill({response});
    });
    await page.evaluate(()=>{lastSync=performance.now()-16000});
    await Promise.race([renewalEntered,new Promise((_,reject)=>setTimeout(()=>reject(new Error('renew did not start')),10000))]);
    const ended=page.waitForResponse(r=>r.url().includes('/api/camera/end?'));
    await page.getByRole('button',{name:'Tắt camera',exact:true}).click(); await ended;
    await page.getByRole('button',{name:'Bật camera',exact:true}).click();
    await page.waitForFunction(()=>running);
    const newer=await page.evaluate(()=>cameraToken);
    release(); await page.waitForTimeout(500);
    const state=await page.evaluate(()=>({running,token:cameraToken,button:document.getElementById('startBtn').textContent}));
    if(!state.running || state.token!==newer || state.button!=='Tắt camera' || errors.length)
      throw new Error(JSON.stringify({state,errors}));
    await page.getByRole('button',{name:'Tắt camera',exact:true}).click();
    console.log(JSON.stringify({passed:true,scope:'delayed old renewal cannot overwrite/stop a new synthetic camera session',pageErrors:errors}));
  } finally {await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
