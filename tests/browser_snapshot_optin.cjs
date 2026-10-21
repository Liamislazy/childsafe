// Manual download only; uses a synthetic canvas camera, never a physical stream.
const {chromium}=require(process.env.CHILDSAFE_PLAYWRIGHT_MODULE || 'playwright');
const url=process.env.CHILDSAFE_TEST_URL || 'http://127.0.0.1:8001';
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({acceptDownloads:true}),errors=[];let downloads=0;
    page.on('pageerror',e=>errors.push(String(e)));page.on('download',()=>downloads++);
    await page.addInitScript(()=>{
      navigator.mediaDevices.getUserMedia=async()=>{
        const c=document.createElement('canvas');c.width=320;c.height=240;
        const ctx=c.getContext('2d');let n=0;
        const timer=setInterval(()=>{ctx.fillStyle='#777';ctx.fillRect(0,0,320,240);ctx.fillStyle='#aaa';ctx.fillRect(n++%300,20,10,10)},50);
        const stream=c.captureStream(20);
        stream.getTracks()[0].addEventListener('ended',()=>clearInterval(timer));
        return stream;
      };
    });
    await page.goto(url);await page.waitForFunction(()=>info!==null);
    if(!await page.locator('#snapshotBtn').isDisabled()) throw new Error('snapshot enabled without a camera');
    await page.getByRole('button',{name:'Bật camera',exact:true}).click();await page.waitForFunction(()=>running);
    if(downloads) throw new Error('automatic image download before user click');
    const pending=page.waitForEvent('download');await page.locator('#snapshotBtn').click();
    const result=await pending;
    if(!/^childsafe_frame_\d+\.png$/.test(result.suggestedFilename())) throw new Error('incorrect image filename');
    await page.getByRole('button',{name:'Tắt camera',exact:true}).click();
    if(!await page.locator('#snapshotBtn').isDisabled() || downloads!==1 || errors.length)
      throw new Error(JSON.stringify({downloads,errors}));
    console.log(JSON.stringify({passed:true,physical_camera_opened:false,scope:'one explicit PNG download, no automatic capture or label'}));
  } finally {await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
