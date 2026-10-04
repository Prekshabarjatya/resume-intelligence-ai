// Resume builder: generate, edit, switch template, live ATS score, print to PDF.
// All user-controlled text goes in through textContent / .value, never innerHTML.
(function () {
  const TEMPLATES = { classic: 'Classic (best for ATS)', modern: 'Modern', compact: 'Compact' };
  const PLACEHOLDER = '[add metric]';
  const LIST_KEYS = [
    ['certifications', 'Certifications'],
    ['achievements', 'Achievements'],
    ['awards', 'Awards'],
    ['publications', 'Publications'],
  ];

  const state = { jobId: null, resume: null, template: 'classic', originalAts: null, atsTimer: null, atsSeq: 0 };

  function h(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else node.setAttribute(k, v);
    }
    (children || []).forEach((c) => c && node.appendChild(c));
    return node;
  }

  const lines = (s) => s.split('\n').map((x) => x.trim()).filter(Boolean);
  const commas = (s) => s.split(',').map((x) => x.trim()).filter(Boolean);

  function emptyResume(r) {
    r.personal_info = r.personal_info || {};
    r.personal_info.links = r.personal_info.links || [];
    [...LIST_KEYS.map(([k]) => k), 'skills', 'experience', 'education', 'projects'].forEach((k) => (r[k] = r[k] || []));
    r.experience.forEach((e) => (e.bullets = e.bullets || []));
    r.projects.forEach((p) => (p.technologies = p.technologies || []));
    return r;
  }

  // ---------- preview ----------
  function section(title, children) {
    return h('section', { class: 'r-section' }, [h('h3', { text: title }), ...children]);
  }

  function withPlaceholders(text) {
    const frag = document.createDocumentFragment();
    text.split(PLACEHOLDER).forEach((part, i, all) => {
      frag.appendChild(document.createTextNode(part));
      if (i < all.length - 1) frag.appendChild(h('mark', { class: 'r-todo', text: PLACEHOLDER }));
    });
    return frag;
  }

  function bulletList(items) {
    const ul = h('ul', { class: 'r-bullets' });
    items.forEach((b) => {
      const li = h('li');
      li.appendChild(withPlaceholders(b));
      ul.appendChild(li);
    });
    return ul;
  }

  function dates(a, b, current) {
    const end = current ? 'Present' : b;
    return [a, end].filter(Boolean).join(' - ');
  }

  function renderPreview() {
    const r = state.resume;
    const info = r.personal_info;
    const sheet = h('div', { class: 'sheet tpl-' + state.template });

    sheet.appendChild(
      h('header', { class: 'r-head' }, [
        h('div', { class: 'r-name', text: info.name || 'Your name' }),
        h('div', { class: 'r-contact', text: [info.email, info.phone, info.location, ...info.links].filter(Boolean).join('  |  ') }),
      ])
    );

    const main = h('main', { class: 'r-main' });
    const aside = h('aside', { class: 'r-aside' });

    if (r.summary) {
      const p = h('p');
      p.appendChild(withPlaceholders(r.summary));
      main.appendChild(section('Summary', [p]));
    }
    if (r.experience.length) {
      main.appendChild(
        section(
          'Experience',
          r.experience.map((e) =>
            h('div', { class: 'r-item' }, [
              h('div', { class: 'r-item-head' }, [
                h('strong', { text: [e.title, e.company].filter(Boolean).join(', ') }),
                h('span', { class: 'r-dates', text: dates(e.start_date, e.end_date, e.is_current) }),
              ]),
              bulletList(e.bullets),
            ])
          )
        )
      );
    }
    if (r.projects.length) {
      main.appendChild(
        section(
          'Projects',
          r.projects.map((p) => {
            const d = h('p');
            d.appendChild(withPlaceholders(p.description || ''));
            return h('div', { class: 'r-item' }, [
              h('div', { class: 'r-item-head' }, [
                h('strong', { text: p.name }),
                h('span', { class: 'r-dates', text: p.technologies.join(', ') }),
              ]),
              d,
            ]);
          })
        )
      );
    }
    if (r.skills.length) aside.appendChild(section('Skills', [h('p', { text: r.skills.join(', ') })]));
    if (r.education.length) {
      aside.appendChild(
        section(
          'Education',
          r.education.map((ed) =>
            h('div', { class: 'r-item' }, [
              h('strong', { text: ed.degree }),
              h('div', { text: [ed.institution, dates(ed.start_date, ed.end_date)].filter(Boolean).join(', ') }),
              ed.details ? h('div', { class: 'r-muted', text: ed.details }) : null,
            ])
          )
        )
      );
    }
    LIST_KEYS.forEach(([key, title]) => {
      if (r[key].length) aside.appendChild(section(title, [bulletList(r[key])]));
    });

    sheet.appendChild(h('div', { class: 'r-body' }, [main, aside]));
    const slot = document.getElementById('rbPreview');
    slot.replaceChildren(sheet);

    const todos = sheet.querySelectorAll('.r-todo').length;
    const note = document.getElementById('rbTodoNote');
    note.textContent = todos ? `${todos} placeholder${todos > 1 ? 's' : ''} to fill before sending: ${PLACEHOLDER}` : '';
    note.hidden = !todos;
  }

  // ---------- ATS ----------
  function renderAts(ats) {
    const box = document.getElementById('rbAts');
    if (!box) return;
    box.replaceChildren();
    if (!ats) {
      box.appendChild(h('div', { class: 'r-muted', text: 'ATS score unavailable right now.' }));
      return;
    }
    const delta = state.originalAts == null ? null : Math.round(ats.score - state.originalAts);
    box.appendChild(h('div', { class: 'rb-ats-score', text: Math.round(ats.score) + '%' }));
    box.appendChild(
      h('div', { class: 'rb-ats-meta' }, [
        h('div', { text: 'ATS score' }),
        h('div', {
          class: 'r-muted',
          text:
            delta == null
              ? `Keyword coverage ${Math.round(ats.keyword_coverage)}%`
              : `${delta >= 0 ? '+' : ''}${delta} vs your original resume. Keyword coverage ${Math.round(ats.keyword_coverage)}%`,
        }),
      ])
    );
    if (ats.issues.length) {
      const ul = h('ul', { class: 'rb-ats-issues' });
      ats.issues.forEach((i) => ul.appendChild(h('li', { text: `${i.severity}: ${i.issue}` })));
      box.appendChild(ul);
    }
  }

  async function refreshAts() {
    if (!state.resume || !document.getElementById('rbAts')) return;
    const seq = ++state.atsSeq;
    try {
      const resp = await fetch(`/analysis/${state.jobId}/ats`, { method: 'POST', body: JSON.stringify(state.resume) });
      if (seq !== state.atsSeq) return;
      renderAts(resp.ok ? (await resp.json()).ats : null);
    } catch (e) {
      if (seq === state.atsSeq) renderAts(null);
    }
  }

  function changed() {
    renderPreview();
    clearTimeout(state.atsTimer);
    state.atsTimer = setTimeout(refreshAts, 500);
  }

  // ---------- editor ----------
  function field(label, value, onInput, opts) {
    opts = opts || {};
    const id = 'rb' + Math.random().toString(36).slice(2, 9);
    const input = opts.rows ? h('textarea', { id, rows: String(opts.rows) }) : h('input', { id, type: 'text' });
    input.value = value || '';
    input.addEventListener('input', () => {
      onInput(input.value);
      changed();
    });
    return h('div', { class: 'rb-field' }, [h('label', { for: id, text: label }), input]);
  }

  function checkbox(label, value, onChange) {
    const id = 'rb' + Math.random().toString(36).slice(2, 9);
    const input = h('input', { id, type: 'checkbox' });
    input.checked = !!value;
    input.addEventListener('change', () => {
      onChange(input.checked);
      changed();
    });
    return h('div', { class: 'rb-field rb-check' }, [input, h('label', { for: id, text: label })]);
  }

  function group(title, children) {
    return h('fieldset', { class: 'rb-group' }, [h('legend', { text: title }), ...children]);
  }

  function removeBtn(onClick) {
    const b = h('button', { type: 'button', class: 'rb-link', text: 'Remove' });
    b.addEventListener('click', onClick);
    return b;
  }

  function addBtn(label, onClick) {
    const b = h('button', { type: 'button', class: 'rb-add', text: label });
    b.addEventListener('click', onClick);
    return b;
  }

  function renderEditor() {
    const r = state.resume;
    const info = r.personal_info;
    const ed = document.getElementById('rbEditor');
    ed.replaceChildren();

    ed.appendChild(
      group('Contact', [
        field('Name', info.name, (v) => (info.name = v)),
        field('Email', info.email, (v) => (info.email = v)),
        field('Phone', info.phone, (v) => (info.phone = v)),
        field('Location', info.location, (v) => (info.location = v)),
        field('Links (one per line)', info.links.join('\n'), (v) => (info.links = lines(v)), { rows: 3 }),
      ])
    );
    ed.appendChild(group('Summary', [field('Summary', r.summary, (v) => (r.summary = v), { rows: 4 })]));
    ed.appendChild(group('Skills', [field('Comma separated', r.skills.join(', '), (v) => (r.skills = commas(v)), { rows: 3 })]));

    const exp = group('Experience', []);
    r.experience.forEach((e, i) => {
      exp.appendChild(
        h('div', { class: 'rb-entry' }, [
          field('Title', e.title, (v) => (e.title = v)),
          field('Company', e.company, (v) => (e.company = v)),
          field('Start', e.start_date, (v) => (e.start_date = v)),
          field('End', e.end_date, (v) => (e.end_date = v)),
          checkbox('I currently work here', e.is_current, (v) => (e.is_current = v)),
          field('Bullets (one per line)', e.bullets.join('\n'), (v) => (e.bullets = lines(v)), { rows: 5 }),
          removeBtn(() => {
            r.experience.splice(i, 1);
            renderEditor();
            changed();
          }),
        ])
      );
    });
    exp.appendChild(
      addBtn('Add experience', () => {
        r.experience.push({ title: '', company: '', start_date: '', end_date: '', is_current: false, bullets: [], source_evidence: '' });
        renderEditor();
        changed();
      })
    );
    ed.appendChild(exp);

    const proj = group('Projects', []);
    r.projects.forEach((p, i) => {
      proj.appendChild(
        h('div', { class: 'rb-entry' }, [
          field('Name', p.name, (v) => (p.name = v)),
          field('Description', p.description, (v) => (p.description = v), { rows: 4 }),
          field('Technologies (comma separated)', p.technologies.join(', '), (v) => (p.technologies = commas(v))),
          removeBtn(() => {
            r.projects.splice(i, 1);
            renderEditor();
            changed();
          }),
        ])
      );
    });
    proj.appendChild(
      addBtn('Add project', () => {
        r.projects.push({ name: '', description: '', technologies: [], source_evidence: '' });
        renderEditor();
        changed();
      })
    );
    ed.appendChild(proj);

    const edu = group('Education', []);
    r.education.forEach((x, i) => {
      edu.appendChild(
        h('div', { class: 'rb-entry' }, [
          field('Degree', x.degree, (v) => (x.degree = v)),
          field('Institution', x.institution, (v) => (x.institution = v)),
          field('Start', x.start_date, (v) => (x.start_date = v)),
          field('End', x.end_date, (v) => (x.end_date = v)),
          field('Details', x.details, (v) => (x.details = v)),
          removeBtn(() => {
            r.education.splice(i, 1);
            renderEditor();
            changed();
          }),
        ])
      );
    });
    edu.appendChild(
      addBtn('Add education', () => {
        r.education.push({ degree: '', institution: '', start_date: '', end_date: '', details: '' });
        renderEditor();
        changed();
      })
    );
    ed.appendChild(edu);

    LIST_KEYS.forEach(([key, title]) => {
      ed.appendChild(group(title, [field('One per line', r[key].join('\n'), (v) => (r[key] = lines(v)), { rows: 4 })]));
    });
  }

  // ---------- shell ----------
  function mount() {
    const root = document.getElementById('builderRoot');
    root.replaceChildren();

    const select = h('select', { id: 'rbTemplate' });
    Object.entries(TEMPLATES).forEach(([k, label]) => select.appendChild(h('option', { value: k, text: label })));
    select.value = state.template;
    select.addEventListener('change', () => {
      state.template = select.value;
      renderPreview();
    });

    const print = h('button', { type: 'button', class: 'btn', text: 'Download PDF' });
    print.addEventListener('click', () => window.print());

    root.appendChild(
      h('div', { class: 'card rb-shell' }, [
        h('h2', { text: 'Your improved resume' }),
        h('p', {
          class: 'r-muted',
          text: 'Rewritten from your own resume for this job. Nothing new is invented: check every line, then edit anything you like.',
        }),
        h('div', { class: 'rb-toolbar' }, [
          h('label', { for: 'rbTemplate', text: 'Template' }),
          select,
          print,
        ]),
        h('div', { id: 'rbAts', class: 'rb-ats' }),
        h('div', { id: 'rbTodoNote', class: 'rb-todo-note', hidden: '' }),
        h('div', { class: 'rb-cols' }, [h('div', { id: 'rbEditor', class: 'rb-editor' }), h('div', { id: 'rbPreview', class: 'rb-preview' })]),
      ])
    );
  }

  async function start(jobId, originalAts, button, status) {
    state.jobId = jobId;
    state.originalAts = originalAts;
    button.disabled = true;
    status.textContent = 'Writing your resume...';
    try {
      const resp = await fetch(`/analysis/${jobId}/resume`, { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `Request failed (${resp.status})`);
      state.resume = emptyResume(data.resume);
      // Compare like with like: the server scores the original resume the same way.
      state.originalAts = data.baseline_ats ? data.baseline_ats.score : null;
      mount();
      renderEditor();
      renderPreview();
      renderAts(data.ats);
      document.getElementById('builderRoot').scrollIntoView({ behavior: 'smooth' });
      status.textContent = '';
    } catch (err) {
      status.textContent = `Could not build the resume: ${err.message}`;
    } finally {
      button.disabled = false;
    }
  }

  window.ResumeBuilder = {
    start,
    clear() {
      clearTimeout(state.atsTimer);
      state.resume = null;
      const root = document.getElementById('builderRoot');
      if (root) root.replaceChildren();
    },
  };
})();
