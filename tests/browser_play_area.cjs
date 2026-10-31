// Render actual page; use synthetic camera/API fixtures only. Never opens a device.
const {chromium}=require(process.env.CHILDSAFE_PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
const base=process.env.CHILDSAFE_TEST_URL || 'http://127.0.0.1:8002';
const fs=require('node:fs');fs.mkdirSync('outputs/play-area',{recursive:true});
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage(),errors=[],external=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>{if(!r.url().startsWith(base)&&!r.url().startsWith('blob:'))external.push(r.url());});
  await page.goto(base+'/play-area');await page.waitForFunction(()=>document.getElementById('model').textContent.includes('Detector:'));
  await page.screenshot({path:'outputs/play-area/initial-ui.png',fullPage:true});
  assert.equal(await page.locator('#items .item').count(),0);
  assert.equal(await page.locator('body').evaluate(el=>el.scrollWidth<=innerWidth),true);
  await page.addInitScript(()=>{
    const cv=document.createElement('canvas');cv.width=800;cv.height=600;let n=0;
    const ctx=cv.getContext('2d');setInterval(()=>{ctx.fillStyle='#aabaca';ctx.fillRect(0,0,800,600);ctx.fillStyle='#223344';ctx.fillRect((n++%50)*5,20,40,300);},50);
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>cv.captureStream(12)});
    Object.defineProperty(navigator.mediaDevices,'enumerateDevices',{value:async()=>[]});
  });
  await page.reload();await page.locator('#auto').uncheck();
  let n=0,qualityGood=true;
  const detection={label:'scissors',name:'Kéo',box:[200,200,270,400],confidence:.85,at_edge:false,action:'Fixture: kiểm tra và cất vật trước khi trẻ tiếp cận.'};
  await page.route('**/api/play-area/scan?*',route=>{n++;return route.fulfill({json:{width:800,height:600,roi:[64,48,736,552],candidates:n<=2?[detection]:[],quality:{reasons:qualityGood?[]:['Ảnh chưa đủ rõ']},detector_ok:true,usable_for_comparison:qualityGood,message:'Dữ liệu mô phỏng kiểm thử giao diện',ms:10}});});
  await page.locator('#camera').click();await page.waitForFunction(()=>document.getElementById('camera').textContent==='Tắt camera');
  await page.locator('#capture').click();await page.waitForSelector('#items .item');
  await page.locator('#items input').check();await page.waitForFunction(()=>document.getElementById('handledCount').textContent==='1');
  await page.locator('#capture').click();await page.waitForFunction(()=>document.getElementById('items').textContent.includes('nhưng máy vẫn phát hiện'));
  assert.equal(await page.locator('#visibleCount').textContent(),'1');
  qualityGood=false;await page.locator('#capture').click();await page.waitForFunction(()=>document.getElementById('items').textContent.includes('Chưa có ảnh đủ rõ'));
  assert.equal(await page.locator('#items input').isChecked(),true);
  qualityGood=true;await page.locator('#capture').click();await page.waitForFunction(()=>document.getElementById('items').textContent.includes('đang chờ lần kiểm tra'));
  await page.locator('#capture').click();await page.waitForFunction(()=>document.getElementById('items').textContent.includes('Không còn phát hiện qua hai ảnh'));
  await page.evaluate(()=>document.getElementById('source').textContent='Ảnh mô phỏng chỉ để kiểm thử giao diện, không phải kết quả model');
  await page.screenshot({path:'outputs/play-area/synthetic-history-ui.png',fullPage:true});
  await page.locator('#camera').click();assert.equal(await page.locator('#visibleCount').textContent(),'—');
  assert.match(await page.locator('#source').textContent(),/đã tắt/);
  // Late response after reset may not repopulate the new session.
  await page.unroute('**/api/play-area/scan?*');
  let release,entered;const seen=new Promise(r=>entered=r);
  await page.route('**/api/play-area/scan?*',async route=>{entered();await new Promise(r=>release=r);try{await route.fulfill({json:{width:800,height:600,roi:[0,0,800,600],candidates:[detection],quality:{reasons:[]},detector_ok:true,usable_for_comparison:true,message:'Late fixture',ms:10}});}catch{}});
  await page.locator('#camera').click();await page.waitForFunction(()=>document.getElementById('camera').textContent==='Tắt camera');
  await page.locator('#capture').click();await seen;await page.locator('#reset').click();release();
  await page.waitForTimeout(100);assert.equal(await page.locator('#items .item').count(),0);
  await page.locator('#camera').click();
  const downloadWait=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadWait;
  await download.saveAs('outputs/play-area/browser-export.json');const report=JSON.parse(fs.readFileSync('outputs/play-area/browser-export.json'));
  assert.equal(report.image_data_included,false);assert.equal(report.items.length,0);
  await page.setViewportSize({width:390,height:844});assert.equal(await page.locator('body').evaluate(el=>el.scrollWidth<=innerWidth),true);
  assert.equal(errors.length,0,errors.join('\n'));assert.equal(external.length,0,external.join('\n'));
  console.log(JSON.stringify({passed:true,physical_camera_opened:false,external_page_requests:external.length,scope:'actual UI, user-confirmation vs detection, poor images, two good misses, stop/reset races, JSON export, mobile layout'}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
