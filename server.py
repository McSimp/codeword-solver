import cherrypy
import json
import simplejson
from cvstuff import CrosswordRecogniserInterface

import cherrypy
from cherrypy.lib.static import serve_file
from trie_structure import Trie
from codeword_solver import CodeBreaker
from codeword_solver import CodeAlphabet

import os.path
cri = CrosswordRecogniserInterface()


class Root:
    @cherrypy.expose
    def index(self, name):
        return serve_file(os.path.join(static_dir, name))

    @cherrypy.expose
    def image(self, filename):
        structure = cri.solve_image(filename)
        return json.dumps(structure)

    @cherrypy.expose
    def solve(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        body = simplejson.loads(rawbody)

        trie = Trie()
        trie.build_trie('wordsEn.txt')

        result = CodeBreaker.solve_code(body, trie, CodeAlphabet([[16, 'c']]))
        dictionary = {}
        for i in range(len(result.code_alphabet)):
            dictionary[i] = result.code_alphabet[i].letter

        return json.dumps(dictionary)

if __name__=='__main__':
    static_dir = os.path.dirname(os.path.abspath(__file__))  # Root static dir is this file's directory.

    cherrypy.config.update( {  # I prefer configuring the server here, instead of in an external file.
            'server.socket_host': '0.0.0.0',
            'server.socket_port': 8080,
        } )
    conf = {
        '/': {  # Root folder.
            'tools.staticdir.on':   True,  # Enable or disable this rule.
            'tools.staticdir.root': static_dir,
            'tools.staticdir.dir':  '',
        }
    }

    cherrypy.quickstart(Root(), '/', config=conf)  # ..and LAUNCH ! :)
