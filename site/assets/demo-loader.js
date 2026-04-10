// Minimal demo-loader: fetch demo-items.json and render simple cards
(function(){
  async function load(url, containerId){
    try{
      const res = await fetch(url, {cache: 'no-store'});
      if(!res.ok) throw new Error('Failed to fetch '+url);
      const items = await res.json();
      const container = document.getElementById(containerId || 'demo-list');
      if(!container) return console.warn('No container found for demo-loader');
      container.innerHTML = '';
      items.forEach(item => container.appendChild(renderCard(item)));
    }catch(e){
      console.error('demo-loader error', e);
    }
  }

  function renderCard(item){
    const card = document.createElement('article');
    card.className = 'demo-card';

    const h = document.createElement('h3');
    h.textContent = item.title || item.id;
    card.appendChild(h);

    if(item.brief){
      const p = document.createElement('p');
      p.className = 'demo-brief';
      p.textContent = item.brief;
      card.appendChild(p);
    }

    if(item.tags && item.tags.length){
      const tagWrap = document.createElement('div');
      tagWrap.className = 'demo-tags';
      item.tags.slice(0,6).forEach(t => {
        const b = document.createElement('span'); b.className='demo-tag'; b.textContent = t; tagWrap.appendChild(b);
      });
      card.appendChild(tagWrap);
    }

    const links = document.createElement('div'); links.className='demo-links';
    if(item.provenance && item.provenance.source_public){
      const a = document.createElement('a');
      a.href = item.provenance.source_public; a.target = '_blank'; a.rel='noopener';
      a.textContent = 'Watch original public video';
      links.appendChild(a);
    }
    if(item.artifact_root){
      const r = document.createElement('a'); r.href = item.artifact_root; r.textContent = 'Browse artifacts'; r.style.marginLeft='12px'; links.appendChild(r);
    }
    card.appendChild(links);

    return card;
  }

  // expose global helper
  window.BonfyreDemoLoader = { load };
})();
