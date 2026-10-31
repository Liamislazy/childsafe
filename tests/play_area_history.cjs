const assert=require('node:assert/strict');
const {PlayAreaHistory}=require('../src/childsafe/web/static/play_area.js');
const c=(x=10)=>({label:'scissors',name:'Kéo',box:[x,20,x+70,150],action:'Kiểm tra'});
const result=(candidates,usable=true)=>({candidates,usable_for_comparison:usable});
const h=new PlayAreaHistory();h.update(result([c()]));
h.items[0].handled=true;
h.update(result([c()]));assert.equal(h.items[0].handled,true);assert.equal(h.items[0].misses,0);
h.update(result([],false));assert.equal(h.items[0].misses,0);assert.equal(h.fresh,false);
h.update(result([]));assert.equal(h.items[0].misses,1);
h.update(result([]));assert.equal(h.items[0].misses,2);
h.update(result([c()]));assert.equal(h.items.length,2);assert.equal(h.items[1].handled,false);
// Ambiguous same-class assignments must not transfer the user's acknowledgement.
const a=new PlayAreaHistory();a.update(result([c(0),c(40)]));a.items[0].handled=true;
a.update(result([c(20)]));assert.equal(a.items.at(-1).handled,false);
a.invalidate();assert.equal(a.fresh,false);
const weak=new PlayAreaHistory();weak.update({...result([c()],false),detector_ok:true});
assert.equal(weak.items.length,1);assert.equal(weak.fresh,false);
weak.update({...result([],false),detector_ok:true});assert.equal(weak.items[0].misses,0);
console.log('Play-area history: no clearance on errors, two good misses, no acknowledgement transfer on ambiguity.');
