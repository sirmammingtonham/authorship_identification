(this.webpackJsonpst_btn_select=this.webpackJsonpst_btn_select||[]).push([[0],{18:function(e,t,n){e.exports=n(26)},25:function(e,t,n){},26:function(e,t,n){"use strict";n.r(t);var c=n(7),r=n.n(c),a=n(16),s=n.n(a),o=n(4),l=n(0),u=n(1),i=n(6),d=n(5),p=n(2),f=n(3),b=n(12),m=(n(25),function(e){Object(p.a)(n,e);var t=Object(f.a)(n);function n(){var e;Object(l.a)(this,n);for(var c=arguments.length,r=new Array(c),a=0;a<c;a++)r[a]=arguments[a];return(e=t.call.apply(t,[this].concat(r))).state={selected:JSON.parse(e.props.args.selected)},e.onClicked=function(t){if(e.state.selected.includes(t)){var n=e.state.selected.filter((function(e){return e!==t}));e.setState((function(e){return{current:t,selected:n}}),(function(){return b.a.setComponentValue(n)}))}else{var c=[].concat(Object(o.a)(e.state.selected),[t]);e.setState((function(e){return{current:t,selected:c}}),(function(){return b.a.setComponentValue(c)}))}},e}return Object(u.a)(n,[{key:"componentDidMount",value:function(){Object(i.a)(Object(d.a)(n.prototype),"componentDidMount",this).call(this),document.body.style.background="transparent"}},{key:"render",value:function(){var e,t,n,c=this,a=this.props.args.words,s=this.state.selected,o=null!==(e=null===(t=this.props)||void 0===t||null===(n=t.theme)||void 0===n?void 0:n.base)&&void 0!==e?e:"light";return r.a.createElement("div",{className:"wrapper"},a.map((function(e,t){return r.a.createElement("button",{onClick:function(){return c.onClicked(t)},className:"".concat(o," ").concat(s.includes(t)?"selected":""),key:t},e)})))}}]),n}(b.b)),h=Object(b.c)(m);s.a.render(r.a.createElement(r.a.StrictMode,null,r.a.createElement(h,null)),document.getElementById("root"))}},[[18,1,2]]]);
//# sourceMappingURL=main.5af2da59.chunk.js.map