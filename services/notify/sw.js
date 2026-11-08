// Shows ChildSafe alerts pushed by the server; no images are ever sent.
self.addEventListener('push', event => {
  const data = event.data ? event.data.json() : {};
  const danger = (data.level || 0) >= 2;
  event.waitUntil(self.registration.showNotification(data.title || 'ChildSafe', {
    body: data.body || '',
    tag: danger ? 'childsafe-danger' : 'childsafe',
    renotify: true,
    requireInteraction: danger,
    vibrate: danger ? [400, 150, 400, 150, 800] : [200],
  }));
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(clients.matchAll({type: 'window'}).then(list =>
    list.length ? list[0].focus() : clients.openWindow('/')));
});
