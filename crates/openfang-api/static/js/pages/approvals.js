// OpenFang Approvals Page — Execution approval queue for sensitive agent actions
'use strict';

function approvalsPage() {
  return {
    approvals: [],
    filterStatus: 'all',
    loading: true,
    loadError: '',
    refreshTimer: null,

    init() {
      var self = this;
      // Seed from app-store cache if fresh (<10s), else fetch once
      var store = (typeof Alpine !== 'undefined') ? Alpine.store('app') : null;
      if (store && store.approvalsCache && store.approvalsCache.length >= 0 && (Date.now() - store.approvalsCacheAt) < 10000) {
        this.approvals = store.approvalsCache;
        this.loading = false;
      } else {
        this.loadData();
      }
      // Subscribe to global poll; no per-page interval
      this._onApprovalsUpdated = function(ev) {
        self.approvals = (ev.detail && ev.detail.approvals) || [];
        self.loading = false;
        self.loadError = '';
      };
      window.addEventListener('openfang:approvals-updated', this._onApprovalsUpdated);
    },

    destroy() {
      if (this._onApprovalsUpdated) {
        window.removeEventListener('openfang:approvals-updated', this._onApprovalsUpdated);
        this._onApprovalsUpdated = null;
      }
      if (this.refreshTimer) {
        clearInterval(this.refreshTimer);
        this.refreshTimer = null;
      }
    },

    get filtered() {
      var f = this.filterStatus;
      if (f === 'all') return this.approvals;
      return this.approvals.filter(function(a) { return a.status === f; });
    },

    get pendingCount() {
      return this.approvals.filter(function(a) { return a.status === 'pending'; }).length;
    },

    async loadData() {
      this.loading = true;
      this.loadError = '';
      try {
        var data = await OpenFangAPI.get('/api/approvals');
        this.approvals = data.approvals || [];
      } catch(e) {
        this.loadError = e.message || 'Could not load approvals.';
      }
      this.loading = false;
    },

    async approve(id) {
      try {
        await OpenFangAPI.post('/api/approvals/' + id + '/approve', {});
        OpenFangToast.success('Approved');
        await this.loadData();
      } catch(e) {
        OpenFangToast.error(e.message);
      }
    },

    async reject(id) {
      var self = this;
      OpenFangToast.confirm('Reject Action', 'Are you sure you want to reject this action?', async function() {
        try {
          await OpenFangAPI.post('/api/approvals/' + id + '/reject', {});
          OpenFangToast.success('Rejected');
          await self.loadData();
        } catch(e) {
          OpenFangToast.error(e.message);
        }
      });
    },

    timeAgo(dateStr) {
      if (!dateStr) return '';
      var d = new Date(dateStr);
      var secs = Math.floor((Date.now() - d.getTime()) / 1000);
      if (secs < 60) return secs + 's ago';
      if (secs < 3600) return Math.floor(secs / 60) + 'm ago';
      if (secs < 86400) return Math.floor(secs / 3600) + 'h ago';
      return Math.floor(secs / 86400) + 'd ago';
    }
  };
}
